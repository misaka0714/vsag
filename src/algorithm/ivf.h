
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "datacell/attribute_bucket_inverted_datacell.h"
#include "datacell/bucket_datacell.h"
#include "datacell/flatten_interface.h"
#include "impl/heap/distance_heap.h"
#include "impl/searcher/basic_searcher.h"
#include "index_common_param.h"
#include "inner_index_interface.h"
#include "ivf_parameter.h"
#include "ivf_partition/ivf_partition_strategy.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "vsag/index.h"

namespace vsag {

// IVF index was introduced since v0.14
class IVF : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    explicit IVF(const IVFParameterPtr& param, const IndexCommonParam& common_param);

    explicit IVF(const ParamPtr& param, const IndexCommonParam& common_param)
        : IVF(std::dynamic_pointer_cast<IVFParameter>(param), common_param){};

    ~IVF() override = default;

    std::vector<int64_t>
    Add(const DatasetPtr& base) override;

    std::string
    AnalyzeIndexBySearch(const vsag::SearchRequest& request) override;

    std::vector<int64_t>
    Build(const DatasetPtr& base) override;

    DatasetPtr
    CalDistanceById(const float* query, const int64_t* ids, int64_t count) const override;

    float
    CalcDistanceById(const float* query, int64_t id) const override;

    void
    Deserialize(StreamReader& reader) override;

    [[nodiscard]] InnerIndexPtr
    ExportModel(const IndexCommonParam& param) const override;

    [[nodiscard]] InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return std::make_shared<IVF>(this->create_param_ptr_, param);
    }

    void
    GetAttributeSetByInnerId(InnerIdType inner_id, AttributeSet* attr) const override;

    void
    GetCodeByInnerId(InnerIdType inner_id, uint8_t* data) const override;

    [[nodiscard]] IndexType
    GetIndexType() override {
        return IndexType::IVF;
    }

    [[nodiscard]] std::string
    GetName() const override {
        return INDEX_IVF;
    }

    [[nodiscard]] int64_t
    GetNumElements() const override;

    void
    GetVectorByInnerId(InnerIdType inner_id, float* data) const override;

    std::string
    GetStats() const override;

    void
    InitFeatures() override;

    [[nodiscard]] DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    void
    Merge(const std::vector<MergeUnit>& merge_units) override;

    [[nodiscard]] DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    [[nodiscard]] DatasetPtr
    SearchWithRequest(const SearchRequest& request) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Train(const DatasetPtr& data) override;

    void
    UpdateAttribute(int64_t id, const AttributeSet& new_attrs) override;

    void
    UpdateAttribute(int64_t id,
                    const AttributeSet& new_attrs,
                    const AttributeSet& origin_attrs) override;

private:
    InnerSearchParam
    create_search_param(const std::string& parameters, const FilterPtr& filter) const;

    template <InnerSearchMode mode = KNN_SEARCH>
    DistHeapPtr
    search(const DatasetPtr& query, const InnerSearchParam& param) const;

    DatasetPtr
    reorder(int64_t topk, DistHeapPtr& input, const float* query) const;

    void
    merge_one_unit(const MergeUnit& unit);

    void
    check_merge_illegal(const MergeUnit& unit) const;

    void
    fill_location_map();

    std::pair<BucketIdType, InnerIdType>
    get_location(InnerIdType inner_id) const;

private:
    BucketInterfacePtr bucket_{nullptr};

    IVFPartitionStrategyPtr partition_strategy_{nullptr};
    BucketIdType buckets_per_data_;

    int64_t total_elements_{0};

    bool is_trained_{false};
    bool use_residual_{false};
    // Training sampling rate parameter
    float train_sample_rate_{1.0f};     // Sampling ratio (0.0-1.0)
    int64_t train_sample_count_{-1};    // Sampling quantity (-1 represents proportional sampling)
    FlattenInterfacePtr reorder_codes_{nullptr};

    std::shared_ptr<SafeThreadPool> thread_pool_{nullptr};

    Vector<uint64_t> location_map_;

    static const uint64_t LOCATION_SPLIT_BIT = 32;
};
}  // namespace vsag

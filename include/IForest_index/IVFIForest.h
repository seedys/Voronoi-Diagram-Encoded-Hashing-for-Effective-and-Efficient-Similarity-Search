#pragma once

#include <faiss/IndexIVF.h>
#include <memory>
#include <vector>

#include "IK/iForest.cpp"

#ifdef USE_256_REG
#include "tools/SimComputer_256reg.h"
#else
#include "tools/SimComputer.h"
#endif

namespace IFlib{

    struct IVFIForest : faiss::IndexIVF
    {
        private:
            // function pointer

            typedef void (IForestMap_nobias::*TransformFunc)(size_t, const float*, uint8_t*) const;
            TransformFunc transform_func;

            void init_functions();

        public:
            std::unique_ptr<IForestMap_nobias> model;
            size_t t;
            size_t psi;
            bool _use_without_replacement;
            int seed;

            IVFIForest(
                faiss::Index *quantizer,
                size_t d,
                size_t nlist,
                // iforest para.
                size_t t,
                size_t psi,
                bool use_without_replacement = false,
                int seed = 0,
                // end
                faiss::MetricType metric = faiss::METRIC_INNER_PRODUCT);

            // void add_core(
            //     faiss::idx_t n,
            //     const float *x,
            //     const faiss::idx_t *xids,
            //     const faiss::idx_t *precomputed_idx,
            //     void *inverted_list_context = nullptr) override;

            void encode_vectors(
                faiss::idx_t n,
                const float *x,
                const faiss::idx_t *list_nos,
                uint8_t *codes,
                bool include_listnos = false) const override;

            void train_encoder(faiss::idx_t n, const float *x, const faiss::idx_t *assign) override;

            faiss::InvertedListScanner *get_InvertedListScanner(
                bool store_pairs = false,
                const faiss::IDSelector *sel = nullptr) const override;
    };

} // namespace IFlib
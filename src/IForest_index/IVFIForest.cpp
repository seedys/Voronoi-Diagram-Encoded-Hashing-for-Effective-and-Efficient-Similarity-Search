#include "IForest_index/IVFIForest.h"

namespace IFlib
{
    namespace
    {
        template <class C, bool use_sel>
        struct IVFIForestScanner : faiss::InvertedListScanner
        {
            using TransformFn = std::vector<uint8_t> (IForestMap_nobias::*)(const float *) const;
            using CmpFn = float (*)(const uint8_t *a, const uint8_t *b, size_t size);

            const IVFIForest *ivf;
            size_t code_size;
            size_t psi;

            TransformFn IFtransformer;
            CmpFn comparator;
            std::vector<uint8_t> xi;
            uint8_t *xi_ptr;

            IVFIForestScanner(const IVFIForest *ivf_, bool store_pairs, const faiss::IDSelector *sel)
                : InvertedListScanner(store_pairs, sel), ivf(ivf_)
            {
                keep_max = true;
                code_size = ivf->code_size;
                psi = ivf->psi;

                if (psi == 2)
                {
                    // 1-bit quantification
                    IFtransformer = &IForestMap_nobias::transform_single_point_packed;
                    comparator = &bit_cmp<float>;
                }
                else if (psi <= 4)
                {
                    // 2-bit quantification
                    IFtransformer = &IForestMap_nobias::transform_single_point_packed_2bit;
                    comparator = &bit2_cmp_v2<float>;
                }
                else if (psi <= 16)
                {
                    // 4-bit quantification
                    IFtransformer = &IForestMap_nobias::transform_single_point_packed_4bit;
                    comparator = &bit4_cmp_v2<float>;
                }
                else
                {
                    // 8-bit quantification
                    IFtransformer = &IForestMap_nobias::transform_single_point;
                    comparator = &uint8_cmp<float>;
                }
            }

            void set_query(const float *query) override
            {
                xi = (ivf->model.get()->*IFtransformer)(query);
                xi_ptr = xi.data();
            }

            void set_list(faiss::idx_t list_no, float coarse_dis) override
            {
                this->list_no = list_no;
            }

            float distance_to_code(const uint8_t *code) const override
            {
                // unused function
                return 0;
            }

            size_t scan_codes(
                size_t list_size,
                const uint8_t *codes,
                const faiss::idx_t *ids,
                float *simi,
                faiss::idx_t *idxi,
                size_t k) const override
            {
                size_t nup = 0;
                for (size_t j = 0; j < list_size; j++)
                {
                    const uint8_t *yj = codes + code_size * j;
                    if (use_sel && !sel->is_member(ids[j]))
                    {
                        continue;
                    }

                    float dis = comparator(xi_ptr, yj, code_size);

                    if (C::cmp(simi[0], dis))
                    {
                        int64_t id = store_pairs ? faiss::lo_build(list_no, j) : ids[j];
                        faiss::heap_replace_top<C>(k, simi, idxi, dis, id);
                        nup++;
                    }
                }
                return nup;
            }
        };

        template <bool use_sel>
        faiss::InvertedListScanner *get_InvertedListScanner1(
            const IVFIForest *ivf,
            bool store_pairs,
            const faiss::IDSelector *sel)
        {
            return new IVFIForestScanner<
                faiss::CMin<float, int64_t>, use_sel>(ivf, store_pairs, sel);
        }
    } // anonymous namespace

    void IVFIForest::init_functions()
    {
        if (psi == 2)
        {
            transform_func = &IForestMap_nobias::transform_packed;
        }
        else if (psi <= 4)
        {
            transform_func = &IForestMap_nobias::transform_packed_2bit;
        }
        else if (psi <= 16)
        {
            transform_func = &IForestMap_nobias::transform_packed_4bit;
        }
        else
        {
            transform_func = &IForestMap_nobias::transform;
        }
    }

    IVFIForest::IVFIForest(
        faiss::Index *quantizer,
        size_t d,
        size_t nlist,
        size_t t,
        size_t psi,
        bool use_without_replacement,
        int seed,
        faiss::MetricType metric)
        : t(t), psi(psi), seed(seed), IndexIVF(quantizer, d, nlist, calculateCodeSize(t, psi), metric)
    {
        init_functions();
        _use_without_replacement = use_without_replacement;

        by_residual = false;
        code_size = calculateCodeSize(t, psi);
    }

    void IVFIForest::train_encoder(faiss::idx_t n, const float *x, const faiss::idx_t *assign)
    {
        std::vector<const float *> data_ptr(n);
        for (size_t i = 0; i < n; i++)
            data_ptr[i] = x + i * d;

        model = std::make_unique<IForestMap_nobias>(data_ptr, t, psi, d, _use_without_replacement, seed);

        if (verbose)
        {
            printf("IForest model trained!\n");
        }
    }

    void IVFIForest::encode_vectors(
        faiss::idx_t n,
        const float *x,
        const faiss::idx_t *list_nos,
        uint8_t *codes,
        bool include_listnos) const {
        // if (_use_1bit_quan)
        // {
        //     model->transform_packed(n, x, codes);
        // }
        // else
        // {
        //     model->transform(n, x, codes);
        // }
        (model.get()->*transform_func)(n, x, codes);
    }

    faiss::InvertedListScanner *IVFIForest::get_InvertedListScanner(
        bool store_pairs,
        const faiss::IDSelector *sel) const
    {
        if (sel)
        {
            return get_InvertedListScanner1<true>(this, store_pairs, sel);
        }
        else
        {
            return get_InvertedListScanner1<false>(this, store_pairs, sel);
        }
    }
} // namespace IFlib
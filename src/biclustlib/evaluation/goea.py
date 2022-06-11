from timeit import default_timer

from goatools.go_enrichment import GOEnrichmentRecord
from goatools.goea.go_enrichment_ns import GOEnrichmentStudy

from ..models import Biclustering, Bicluster


def _get_gene_set(bicluster: Bicluster, data_index: list[str]) -> list[str]:
    return [data_index[i] for i in bicluster.rows]


def _get_most_enriched_term(goea_study: GOEnrichmentStudy, gene_set: list[str]) -> GOEnrichmentRecord:
    return min(goea_study.run_study(gene_set, prt=None), key=lambda nt: nt.p_fdr_bh)


def enrich_biclustering(goea_study: GOEnrichmentStudy,
                        data_index: list[str],
                        biclustering: Biclustering) -> dict:
    start = default_timer()
    result = [_get_most_enriched_term(goea_study, _get_gene_set(bicluster, data_index))
              for bicluster in biclustering.biclusters]
    end = default_timer()
    return {'enrichment_time': end - start,
            'enriched_biclustering': result}


def enrichment_rate(enriched_biclustering: list[GOEnrichmentRecord],
                    alpha: float) -> float:
    return len(list(filter(lambda r: r.p_fdr_bh <= alpha, enriched_biclustering))) / max(1, len(enriched_biclustering))

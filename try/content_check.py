import torch
from sentence_transformers import SentenceTransformer, util
def _get_specter_model():
    print(f"Initializing baseline Embedding Model ('allenai/specter2_base')...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _specter_model = SentenceTransformer("allenai/specter2_base", device=device)
    return _specter_model 


def content_detect(title: str, abstract: str, st_model) -> bool:
    """
    Compute the similarity of the title and the abstract. Returns True if it matches semantic expectations (>= 0.50).
    """
    # Execute semantic alignment validation
    if st_model is not None:
        try:
            t_emb = st_model.encode(title.strip())
            a_emb = st_model.encode(abstract.strip())
            similarity_score = util.cos_sim(t_emb, a_emb).item()
            print(f"Semantic similarity score between title and abstract: {similarity_score:.4f}")
            if similarity_score < 0.5:
                return False
        except Exception:
            return False  # Fail-safe reject on vectorization error
    return True


title="Wet Digestion of Plant Material Gives Low Boron Values"
abstract="ADVERTISEMENT RETURN TO ISSUEPREVArticleNEXTWet Digestion of Plant Material Gives Low Boron ValuesJ. T. HatcherCite this: Anal. Chem. 1960, 32, 6, 726Publication Date (Print):May 1, 1960Publication History Published online1 May 2002Published inissue 1 May 1960https://pubs.acs.org/doi/10.1021/ac60162a053https://doi.org/10.1021/ac60162a053research-articleACS PublicationsRequest reuse permissionsArticle Views67Altmetric-Citations1LEARN ABOUT THESE METRICSArticle Views are the COUNTER-compliant sum of full text article downloads since November 2008 (both PDF and HTML) across all institutions and individuals. These metrics are regularly updated to reflect usage leading up to the last few days.Citations are the number of other articles citing this article, calculated by Crossref and updated daily. Find more information about Crossref citation counts.The Altmetric Attention Score is a quantitative measure of the attention that a research article has received online. Clicking on the donut icon will load a page at altmetric.com with additional details about the score and the social media presence for the given article. Find more information on the Altmetric Attention Score and how the score is calculated. Share Add toView InAdd Full Text with ReferenceAdd Description ExportRISCitationCitation and abstractCitation and referencesMore Options Share onFacebookTwitterWechatLinked InRedditEmail Other access optionsGet e-Alertsclose Get e-Alerts"


match= content_detect(title, abstract, st_model=_get_specter_model())
if match==False:
    content_reason = "Semantic mismatch between title and abstract."
else:
    content_reason = "Semantic match between title and abstract."
print(f"Semantic Match: {match} | Reason: {content_reason}")

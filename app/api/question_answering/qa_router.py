from fastapi import APIRouter
from app.api.question_answering.user_index import qa, table_qa, llama_qa
from app.api.question_answering.anc_index import anc_qa
from app.api.question_answering.summarization import summary
from app.api.question_answering.generation import generate_email
from app.api.question_answering.faq import askken_faq
from app.api.question_answering.web import web_search

router = APIRouter(tags=["Question Answering System"])

router.include_router(qa.router)
router.include_router(llama_qa.router)
router.include_router(table_qa.router)
router.include_router(anc_qa.router)
router.include_router(summary.router)
router.include_router(generate_email.router)
router.include_router(askken_faq.router)
router.include_router(web_search.router)
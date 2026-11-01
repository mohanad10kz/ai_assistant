from prompts import AGENT_INSTRUCTIONS


def test_named_websites_are_opened_directly() -> None:
    assert "إذا أشار المستخدم إلى موقع إلكتروني أو خدمة أو نطاق" in AGENT_INSTRUCTIONS
    assert "استخدم search_the_web فقط عندما لا يتم تحديد موقع" in AGENT_INSTRUCTIONS

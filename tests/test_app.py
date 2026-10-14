from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]

def test_bundled_samples_and_clahe():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value=='SkinLens'
    for sample in app.radio[0].options:
        app.radio[0].set_value(sample).run()
        assert not app.exception
        assert len(app.metric)==4
    app.checkbox[0].check().run()
    assert not app.exception
    assert len(app.metric)==4

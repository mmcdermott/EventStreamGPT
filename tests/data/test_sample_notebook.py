from pytest_notebook.nb_regression import NBRegressionFixture

fixture = NBRegressionFixture(exec_timeout=50, exec_cwd="/home/mmd/Projects/EventStreamGPT/sample_data")
fixture.diff_color_words = False
fixture

result = fixture.check(
    "/home/mmd/Projects/EventStreamGPT/sample_data/examine_synthetic_data.ipynb", raise_errors=True
)

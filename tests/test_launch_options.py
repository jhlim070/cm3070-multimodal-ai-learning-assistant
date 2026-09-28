"""
Tests filtering of optional Gradio launch arguments.

The launch configuration contains options that may not be supported by every
Gradio version. Supported options should be passed independently so that an
unsupported option does not prevent valid options, such as the 50 MB upload
limit, from being applied.
"""
import inspect
import re

import app
import validation


def test_launch_options_are_filtered_one_by_one():
    def launch(self=None, css=None, max_file_size=None):
        pass
    got = app.build_launch_kwargs(launch)
    assert got == {"css": app.RATING_PANEL_CSS, "max_file_size": "50mb"}
    assert "show_api" not in got


def test_unsupported_option_does_not_remove_supported_options():
    def launch(max_file_size=None, show_api=None):
        pass
    assert set(app.build_launch_kwargs(launch)) == {"max_file_size", "show_api"}

    def only_css(css=None):
        pass
    assert set(app.build_launch_kwargs(only_css)) == {"css"}

    def nothing():
        pass
    assert app.build_launch_kwargs(nothing) == {}


def test_uninspectable_launch_function_returns_no_options():
    assert app.build_launch_kwargs(object()) == {}


def test_the_upload_cap_is_kept_whenever_this_gradio_supports_it():
    accepted = set(inspect.signature(app.demo.launch).parameters)
    got = app.build_launch_kwargs(app.demo.launch)
    for option in app.WANTED_LAUNCH_OPTIONS:
        assert (option in got) == (option in accepted), option
    if "max_file_size" in accepted:
        assert got["max_file_size"] == "50mb"


def test_the_server_side_cap_matches_the_validators_limit():
    max_mb = int(re.match(r"(\d+)mb", app.WANTED_LAUNCH_OPTIONS["max_file_size"]).group(1))
    assert max_mb * 1024 * 1024 == validation.MAX_FILE_BYTES


def test_rating_panel_stylesheet_is_included_in_launch_options():
    assert ".rating-prompt" in app.WANTED_LAUNCH_OPTIONS["css"]

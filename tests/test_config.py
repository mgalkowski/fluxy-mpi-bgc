def test_import():
    from fluxy.config import initialize_settings


def test_initialize_settings():
    from fluxy.config import initialize_settings

    initialize_settings()


def test_model_colors():

    # Legacy function
    # This function should be changed, as the color model should be somehow cleaner
    from fluxy.config import set_model_colors

    model_colors = set_model_colors(
        models=["intem_name_edgar", "elris_name_edgar", "rhime_name_edgar"],
        model_colors={
            "intem": [["blue", "dodgerblue"], ["dodgerblue", "skyblue"]],
            "elris": [
                ["purple", "mediumpurple"],
                ["deeppink", "pink"],
                ["darkorange", "red"],
            ],
            "rhime": [
                ["darkgreen", "green"],
                ["limegreen", "palegreen"],
                ["olive", "lightgreen"],
            ],
        },
    )

    

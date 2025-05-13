from src.data_manager.CoordinateManager import CoordinateManager


def test_edge_to_dict_edge():
    # Completely symmetrical edge case.
    assert CoordinateManager.edge_to_dict_edge(5) == {"X_left": 5, "X_right": 5,
                                                      "Y_bottom": 5, "Y_top": 5}

    # Partially symmetrical edge case, horizontal relative to vertical.
    assert CoordinateManager.edge_to_dict_edge((3, 2)) == {"X_left": 3, "X_right": 3,
                                                           "Y_bottom": 2, "Y_top": 2}

    # Completely asymmetrical edge case.
    assert CoordinateManager.edge_to_dict_edge(((7, 8), (4, 9))) == {"X_left": 7, "X_right": 8,
                                                                     "Y_bottom": 4, "Y_top": 9}

    # Partially symmetrical vertical and asymmetrical horizontal edge case.
    assert CoordinateManager.edge_to_dict_edge(((7, 8), 12)) == {"X_left": 7, "X_right": 8,
                                                                 "Y_bottom": 12, "Y_top": 12}

    # Partially symmetrical horizontal and asymmetrical vertical edge case.
    assert CoordinateManager.edge_to_dict_edge((1, (4, 9))) == {"X_left": 1, "X_right": 1,
                                                                "Y_bottom": 4, "Y_top": 9}


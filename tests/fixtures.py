"""
This module contains all fixtures functions for the examples with calabru.
"""
import ospgrillage as og
import pycba as cba


def beam_with_patch_load(p_matrix: list):
    """
    Simple grillage model created in ospgrillage, with patch load applied onto the entire surface of the grillage.
    The updating parameters are: (1) the magnitude of the patch load vertex, and (2) the second moment of area, I, of
    the interior I-beam girders (longitudinal members).

    :param p_matrix: A list of float/int of the updating parameters.
    :type p_matrix: list
    :return: A list of response measurement of the model. In this function, the response contains two element forces
     in two elements i.e. ele 23 and 25.

    """
    # sort parameter variable
    P = p_matrix[0]
    Iz = p_matrix[1]

    # Adopted units: N and m
    kilo = 1e3
    milli = 1e-3
    N = 1
    m = 1
    mm = milli * m
    m2 = m**2
    m3 = m**3
    m4 = m**4
    kN = kilo * N
    MPa = N / ((mm) ** 2)
    GPa = kilo * MPa

    # parameters of bridge grillage
    L = 10 * m  # span
    w = 5 * m  # width
    n_l = 7  # number of longitudinal members
    n_t = 10  # number of transverse members
    edge_dist = 1 * m  # distance between edge beam and first exterior beam
    ext_to_int_dist = (
        2.2775 * m
    )  # distance between first exterior beam and first interior beam
    angle = 0  # skew angle
    mesh_type = "Oblique"

    # define material
    concrete = og.create_material(
        material="concrete", code="AS5100-2017", grade="65MPa"
    )

    # define sections (parameters from LUSAS model)
    edge_longitudinal_section = og.create_section(
        A=0.934 * m2,
        J=0.1857 * m3,
        Iz=0.3478 * m4,
        Iy=0.213602 * m4,
        Az=0.444795 * m2,
        Ay=0.258704 * m2,
    )
    0.3694
    longitudinal_section = og.create_section(
        A=1.025 * m2,
        J=0.1878 * m3,
        Iz=Iz * m4,
        Iy=0.113887e-3 * m4,
        Az=0.0371929 * m2,
        Ay=0.0371902 * m2,
    )

    transverse_section = og.create_section(
        A=0.504 * m2,
        J=5.22303e-3 * m3,
        Iy=0.32928 * m4,
        Iz=1.3608e-3 * m4,
        Ay=0.42 * m2,
        Az=0.42 * m2,
    )

    end_transverse_section = og.create_section(
        A=0.504 / 2 * m2,
        J=2.5012e-3 * m3,
        Iy=0.04116 * m4,
        Iz=0.6804e-3 * m4,
        Ay=0.21 * m2,
        Az=0.21 * m2,
    )

    # define grillage members
    longitudinal_beam = og.create_member(
        section=longitudinal_section, material=concrete
    )
    edge_longitudinal_beam = og.create_member(
        section=edge_longitudinal_section, material=concrete
    )
    transverse_slab = og.create_member(section=transverse_section, material=concrete)
    end_transverse_slab = og.create_member(
        section=end_transverse_section, material=concrete
    )

    # create grillage
    simple_bridge = og.create_grillage(
        bridge_name="simple_bridge",
        long_dim=L,
        width=w,
        skew=angle,
        num_long_grid=n_l,
        num_trans_grid=n_t,
        edge_beam_dist=edge_dist,
        mesh_type=mesh_type,
    )

    simple_bridge.set_member(longitudinal_beam, member="interior_main_beam")
    simple_bridge.set_member(longitudinal_beam, member="exterior_main_beam_1")
    simple_bridge.set_member(longitudinal_beam, member="exterior_main_beam_2")
    simple_bridge.set_member(edge_longitudinal_beam, member="edge_beam")
    simple_bridge.set_member(transverse_slab, member="transverse_slab")
    simple_bridge.set_member(end_transverse_slab, member="start_edge")
    simple_bridge.set_member(end_transverse_slab, member="end_edge")
    simple_bridge.create_osp_model(pyfile=False)

    # add load case
    # Patch load over entire bridge deck (P is kN/m2)
    P = P * kN / m2  # magnitude of patch vertex
    patch_point_1 = og.create_load_vertex(x=0, z=0, p=P)
    patch_point_2 = og.create_load_vertex(x=L, z=0, p=P)
    patch_point_3 = og.create_load_vertex(x=L, z=w, p=P)
    patch_point_4 = og.create_load_vertex(x=0, z=w, p=P)
    test_patch_load = og.create_load(
        loadtype="patch",
        name="Test Load",
        point1=patch_point_1,
        point2=patch_point_2,
        point3=patch_point_3,
        point4=patch_point_4,
    )

    test_point_load = og.create_load(
        loadtype="point",
        name="Test Load",
        point1=og.create_load_vertex(x=L / 2, z=w / 2, p=P),
    )

    # Create load case, add loads, and assign
    patch_case = og.create_load_case(name="test patch load case")
    patch_case.add_load(test_patch_load)
    point_case = og.create_load_case(name="test point load case")
    point_case.add_load(test_point_load)
    # sn8474.add_load_case(patch_case)
    simple_bridge.add_load_case(point_case)

    simple_bridge.analyze()
    results = simple_bridge.get_results()

    # arbitrary force components
    r_mat = [og.ops.eleForce(20)[3], og.ops.eleForce(25)[1]]

    return r_mat


def beam_with_patch_load_including_deflected_shape_output(p_matrix: list):
    """
    A version of beam_with_patch_load - this time the output measurements comprise the deflected shape across the node
    elements.

    :param p_matrix: A list of float/int of the updating parameters.
    :type p_matrix: list
    :return: A list of response measurement of the model. In this function, the response contains (1) ele force of ele
    23, and (2) the deflected shape across all elements

    """
    # sort parameter variable
    # P = p_matrix[0]
    P = 1000
    Iz = p_matrix[0]

    # Adopted units: N and m
    kilo = 1e3
    milli = 1e-3
    N = 1
    m = 1
    mm = milli * m
    m2 = m**2
    m3 = m**3
    m4 = m**4
    kN = kilo * N
    MPa = N / ((mm) ** 2)
    GPa = kilo * MPa

    # parameters of bridge grillage
    L = 10 * m  # span
    w = 5 * m  # width
    n_l = 7  # number of longitudinal members
    n_t = 10  # number of transverse members
    edge_dist = 1 * m  # distance between edge beam and first exterior beam
    ext_to_int_dist = (
        2.2775 * m
    )  # distance between first exterior beam and first interior beam
    angle = 0  # skew angle
    mesh_type = "Oblique"

    # define material
    concrete = og.create_material(
        material="concrete", code="AS5100-2017", grade="65MPa"
    )

    # define sections (parameters from LUSAS model)
    edge_longitudinal_section = og.create_section(
        A=0.934 * m2,
        J=0.1857 * m3,
        Iz=0.3478 * m4,
        Iy=0.213602 * m4,
        Az=0.444795 * m2,
        Ay=0.258704 * m2,
    )

    longitudinal_section = og.create_section(
        A=1.025 * m2,
        J=0.1878 * m3,
        Iz=Iz * m4,
        Iy=0.113887e-3 * m4,
        Az=0.0371929 * m2,
        Ay=0.0371902 * m2,
    )

    transverse_section = og.create_section(
        A=0.504 * m2,
        J=5.22303e-3 * m3,
        Iy=0.32928 * m4,
        Iz=1.3608e-3 * m4,
        Ay=0.42 * m2,
        Az=0.42 * m2,
    )

    end_transverse_section = og.create_section(
        A=0.504 / 2 * m2,
        J=2.5012e-3 * m3,
        Iy=0.04116 * m4,
        Iz=0.6804e-3 * m4,
        Ay=0.21 * m2,
        Az=0.21 * m2,
    )

    # define grillage members
    longitudinal_beam = og.create_member(
        section=longitudinal_section, material=concrete
    )
    edge_longitudinal_beam = og.create_member(
        section=edge_longitudinal_section, material=concrete
    )
    transverse_slab = og.create_member(section=transverse_section, material=concrete)
    end_transverse_slab = og.create_member(
        section=end_transverse_section, material=concrete
    )

    # create grillage
    simple_bridge = og.create_grillage(
        bridge_name="simple_bridge",
        long_dim=L,
        width=w,
        skew=angle,
        num_long_grid=n_l,
        num_trans_grid=n_t,
        edge_beam_dist=edge_dist,
        mesh_type=mesh_type,
    )

    simple_bridge.set_member(longitudinal_beam, member="interior_main_beam")
    simple_bridge.set_member(longitudinal_beam, member="exterior_main_beam_1")
    simple_bridge.set_member(longitudinal_beam, member="exterior_main_beam_2")
    simple_bridge.set_member(edge_longitudinal_beam, member="edge_beam")
    simple_bridge.set_member(transverse_slab, member="transverse_slab")
    simple_bridge.set_member(end_transverse_slab, member="start_edge")
    simple_bridge.set_member(end_transverse_slab, member="end_edge")
    simple_bridge.create_osp_model(pyfile=False)

    # add load case
    # Patch load over entire bridge deck (P is kN/m2)
    P = P * kN / m2  # magnitude of patch vertex
    patch_point_1 = og.create_load_vertex(x=0, z=0, p=P)
    patch_point_2 = og.create_load_vertex(x=L, z=0, p=P)
    patch_point_3 = og.create_load_vertex(x=L, z=w, p=P)
    patch_point_4 = og.create_load_vertex(x=0, z=w, p=P)
    test_patch_load = og.create_load(
        loadtype="patch",
        name="Test Load",
        point1=patch_point_1,
        point2=patch_point_2,
        point3=patch_point_3,
        point4=patch_point_4,
    )

    test_point_load = og.create_load(
        loadtype="point",
        name="Test Load",
        point1=og.create_load_vertex(x=L / 2, z=w / 2, p=P),
    )

    # Create load case, add loads, and assign
    patch_case = og.create_load_case(name="test patch load case")
    patch_case.add_load(test_patch_load)
    point_case = og.create_load_case(name="test point load case")
    point_case.add_load(test_point_load)
    # sn8474.add_load_case(patch_case)
    simple_bridge.add_load_case(point_case)

    simple_bridge.analyze()
    results = simple_bridge.get_results()

    # arbitrary force components
    r_mat = [[og.ops.nodeDisp(n)[1] for n in [2, 9, 16, 23, 30, 37, 44, 51, 58, 65]]]
    # og.ops.eleForce(25)[1],
    return r_mat


def pycba_example(p_matrix: list):
    """
    Simple beam example of pycba - https://ccaprani.github.io/pycba/notebooks/intro.html.

    A simple two span pycba model is set up to update the force magnitude of udl,P, on span 1.

    :param p_matrix: A list of float/int of the updating parameters.
    :type p_matrix: list
    :return: A list of response measurement of the model. In this function, the response is the maximum bending moment
    at midspan of span 1
    """
    LM = [[1, 1, p_matrix[0], 0, 0], [2, 1, 20, 0, 0]]
    EI = 30 * 600e7 * 1e-6

    L = [7.5, 7.0]
    R = [-1, 0, -1, 0, -1, 0]

    beam_analysis = cba.BeamAnalysis(L, EI, R, LM)
    beam_analysis.analyze()
    beam_analysis.plot_results()

    r_mat = [beam_analysis.beam_results.vRes[1].M.max()]
    return r_mat


def pycba_example_find_ei(p_matrix: list):
    """
    A more realistic example of the simple pycba beam example - this time with bending stiffness EI of beam and
    displacements along span 1 (quarter, mid and three quarter) as updating parameter and response measurement
    respectively.

    The loading in this example is a stationary vehicle positioned at a = 6m of span 1.

    """
    # create 5 axle semi trailer at span 1
    N = 1
    tonne = 9810 * N
    m = 1
    gross_veh_w = 45 * tonne
    a = 6 * m
    LM = [
        [1, 2, 7 * tonne, a, 0],
        [1, 2, 8 * tonne, a + 3.75, 0],
        [1, 2, 10 * tonne, a + 10.25, 0],
        [1, 2, 10 * tonne, a + 11.45, 0],
        [1, 2, 10 * tonne, a + 12.65, 0],
    ]

    # create pycba model
    L = [20, 20]  # span distance
    EI = p_matrix[0]  # * og.np.ones(len(L))
    R = [-1, 0, -1, 0, -1, 0]  # restraint for two spans

    # static vehicle method
    beam_analysis = cba.BeamAnalysis(L, EI, R, LM)
    beam_analysis.analyze()

    # extract displacements at quarter, mid and thirds of span 1. Here the indexes are based on the D list size of 103.
    r_mat = [
        beam_analysis.beam_results.vRes[0].D[26],
        beam_analysis.beam_results.vRes[0].D[52],
        beam_analysis.beam_results.vRes[0].D[78],
    ]
    return r_mat


def pycba_example_moving_veh_ei(p_matrix: list):
    """
    A further refined example of pycba_example_find_ei - this time the load is traversing across the bridge.

    The updating paramter remains the same (EI). The response parameter are time history of displacements extracted
    at midspan of span 1 for each position of the moving load traverse. The updating procedure aims to minimize the
    error between a known target response history and the response history corresponding to the starting updating
    parameter.

    """
    extract_no = 100
    N = 1
    tonne = 9810 * N
    m = 1
    gross_veh_w = 45 * tonne
    a = 6 * m
    axle_spacings = og.np.array([3.75, 6.5, 1.2, 1.2])
    axle_weights = og.np.array(
        [7 * tonne, 8 * tonne, 10 * tonne, 10 * tonne, 10 * tonne]
    )

    L = [20, 20]  # span distance
    EI = p_matrix[0] * 1e11 * og.np.ones(len(L)) * 1e-6
    R = [-1, 0, -1, 0, -1, 0]  # restraint for two spans

    bridge_analysis = cba.BridgeAnalysis()
    bridge = bridge_analysis.add_bridge(L, EI, R)
    vehicle = bridge_analysis.add_vehicle(axle_spacings, axle_weights)

    env = bridge_analysis.run_vehicle(0.5, plot_env=False, plot_all=False)
    # extract moment at 100th position along the 206 plotting position (near midspan of first span)
    moment = [value.results.D[extract_no] for value in env.vResults]
    return moment


if __name__ == "__main__":
    # b = pycba_example(p_matrix=[10])
    # b = pycba_example_moving_veh_ei(p_matrix=[10])

    # b = beam_with_patch_load_including_deflected_shape_output(p_matrix=[1000, 0.369])
    b = beam_with_patch_load(p_matrix=[1000, 0.369])
    print(b)
    print("exec main")

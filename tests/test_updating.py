"""
Main module for model calibration
"""
import pytest
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath("../"))
from calibrator import *
import fixtures


def test_basic_update():
    """
    This test checks the basic updating procedure. The test uses creates an ospgrillage model, updates
    two parameters [P, I], using two pretend measurements of forces as targets.

    """
    start = [2000, 0.2]
    target = [1558.6480741602825, 2956.470189168508]  # [-30785, -3801]
    simple_beam_updating = ModelUpdating(
        function_handle=fixtures.beam_with_patch_load,
        param_list=start,
        target_list=target,
    )
    simple_beam_updating.update_model()
    print(simple_beam_updating.param_update_history)
    print(simple_beam_updating.response_history)
    tol = 1  # custom tolerance
    tol2 = 1e-3
    assert np.isclose(simple_beam_updating.param_update_history[-1][0], 1000, rtol=tol)
    assert np.isclose(
        simple_beam_updating.param_update_history[-1][1], 0.3694, rtol=tol2
    )


def test_target_resp_as_list():
    """
    Checks updating using a list of measurements as an element entry of the target measurement. The updating should
    consider the root mean square between the target measurement and metrics from function handler.

    """
    start = [ 0.2]
    target = [[0.0, 5.2433545467515094e-05, 9.984279414955342e-05, 0.00013613326490916162, 0.00015584130641133412, 0.0001558413064113349, 0.00013613326490916387, 9.984279414955602e-05, 5.243354546751693e-05, 0.0]]  # [-30785, -3801]
    simple_beam_updating = ModelUpdating(
        function_handle=fixtures.beam_with_patch_load_including_deflected_shape_output,
        param_list=start,
        target_list=target,
        max_error=0.1,
    )
    simple_beam_updating.update_model()
    print(simple_beam_updating.param_update_history)
    tol = 1  # custom tolerance
    tol2 = 1e-1
    #assert np.isclose(simple_beam_updating.param_update_history[-1][0], 1000, rtol=tol)
    assert np.isclose(
        simple_beam_updating.param_update_history[-1][0], 0.3694, rtol=tol2
    )


def test_more_param_than_response():
    """
    Checks inversion of sensitivity matrix for updating case where there are more parameters than responses.

    :return:
    """
    pass


def test_robust():
    """
    Test robustness of calibrator updating a pycba model.

    Point load of 20 N gives a midspan bending moment of 65 Nm.

    :return:
    """
    start = [10]  # P magnitude of UDL in span 1
    target = [65.42524]
    pycba_updating = ModelUpdating(
        function_handle=fixtures.pycba_example, param_list=start, target_list=target
    )
    pycba_updating.update_model()
    tol = 1e-3  # tolerance for P
    assert np.isclose(pycba_updating.param_update_history[-1][0], 20, rtol=tol)


def test_static_truck_pycba():
    """
    A more realistic test of the pycba example entailing a semi trailer vehicle - updating the bridge's EI to match
    known displacements measurements.

    Test pass if the target EI value is achieved in updating.

    The pretend target measurements are deflections measured at quarter, midspan, and thirds of Span 1. White noise
    are added to the measurements which correspond to values at EI  = 180000
    """
    start_ei = [20e3]  # starting value of updating parameter EI
    # target measurement with added white noise (np.rand)
    target_def = [
        -107.00412246501661 + np.random.rand(1)[0],
        -149.1317283092082 + np.random.rand(1)[0],
        -96.10790029773669 + np.random.rand(1)[0],
    ]
    pycba_updating = ModelUpdating(
        function_handle=fixtures.pycba_example_find_ei,
        param_list=start_ei,
        target_list=target_def,
    )
    pycba_updating.update_model()
    tol = 1e2
    end_ei = 30 * 600e7 * 1e-6  # known target EI
    print(pycba_updating.param_update_history)
    assert np.isclose(pycba_updating.param_update_history[-1][0], end_ei, rtol=tol)


def test_moving_truck_pycba():
    """
    This is a further realistic test of the pycba example entailing a semi trailer vehicle - updating the bridge's EI to match
    known displacements measurements.

    Test pass if the target EI value is achieved in updating.

    The target known measurements is the time history of the moving vehicle at EI input = 10

    """
    start_ei = [2]  # starting value of updating parameter EI
    # target measurement with added white noise (np.rand) - known EI of 1e12
    known_measurement = [
        0.0,
        -0.0018260372856796626,
        -0.0023223769104375696,
        -0.004100979213351525,
        -0.004609820533500494,
        -0.006323521209960688,
        -0.006827397581813071,
        -0.008458729988132756,
        -0.010179646049882897,
        -0.013026086871327582,
        -0.014771167040038712,
        -0.017480344425467656,
        -0.01913811689707078,
        -0.02167260232429528,
        -0.02320563857660074,
        -0.025528003523438474,
        -0.026898875034257155,
        -0.028971690978515673,
        -0.03014296922566336,
        -0.03192880764515603,
        -0.032863064106445114,
        -0.03587383558133831,
        -0.03817732089577046,
        -0.041797961502632625,
        -0.04541029098774796,
        -0.04933652266158317,
        -0.054410164296384865,
        -0.059442287215873406,
        -0.06376063766229838,
        -0.06814646924401821,
        -0.07183105556939658,
        -0.07545831024679509,
        -0.07839684688456622,
        -0.08115323909106979,
        -0.08323344047466831,
        -0.08500668464371927,
        -0.08611626520658294,
        -0.08679407577161342,
        -0.08682074994718116,
        -0.0862908413416415,
        -0.08513605756334651,
        -0.08356199364930407,
        -0.08165727835107067,
        -0.07937240784885913,
        -0.0767010423228634,
        -0.07359367795328683,
        -0.07004397492033865,
        -0.0660024294042192,
        -0.061587455646361366,
        -0.0569968472965995,
        -0.052248793413151695,
        -0.04728611311546796,
        -0.04211048552300251,
        -0.036664729755210644,
        -0.03095052493154058,
        -0.024910690171451023,
        -0.018546904594392493,
        -0.011801987319817533,
        -0.004677617467181343,
        0.0028833858440644846,
        0.010879343494466797,
        0.019211493788028368,
        0.027569035768267167,
        0.03593454855759581,
        0.04397909206930803,
        0.051630076422223656,
        0.05868080355872861,
        0.06490748619290437,
        0.07033171701964343,
        0.07498060589730347,
        0.07888126268424421,
        0.08206079723882495,
        0.08454631941940886,
        0.08636493908434888,
        0.08754376609200942,
        0.08810991030074744,
        0.08809048156892293,
        0.08751258975489735,
        0.08640334471702558,
        0.08478985631367002,
        0.08269923440318966,
        0.08128244308326366,
        0.07943852088385435,
        0.07719036057424046,
        0.07456085492370353,
        0.07157289670152979,
        0.06824937867699418,
        0.06461319361938132,
        0.06132973794108711,
        0.058419494646208756,
        0.055258033330533515,
        0.05186342723296755,
        0.048253749592416954,
        0.04444707364778708,
        0.04046147263798709,
        0.03631501980192017,
        0.03202578837849421,
        0.027611851606615383,
        0.023091282725189677,
        0.018482154973123416,
        0.01380254158932254,
        0.00987364536658996,
        0.006710527337347946,
        0.0036827567820682233,
        0.0020862572232845968,
        0.0004819259276416346,
    ]

    known_measurement_with_noise = [a + np.random.rand(1)[0] for a in known_measurement]

    known_target_ei = 100
    pycba_updating = ModelUpdating(
        function_handle=fixtures.pycba_example_moving_veh_ei,
        param_list=start_ei,
        target_list=[known_measurement_with_noise],
    )

    pycba_updating.update_model()
    tol = 1
    print(pycba_updating.param_update_history)
    assert np.isclose(
        pycba_updating.param_update_history[-1][0], known_target_ei, rtol=tol
    )

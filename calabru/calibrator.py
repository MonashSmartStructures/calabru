# -*- coding: utf-8 -*-
"""
Main module for model calibration / updating procedures.
"""
import numpy as np

from datetime import datetime
import copy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class ModelUpdating:
    """
    Main updating/calibrating class.
    """

    def __init__(self, function_handle, **kwargs):
        """
        :param function_handle: Function handler to create and analyze models in Python environment. Note the function
         handler must be able to pass in updating parameters and return desirable responses.
        :param sensitivity_type: Method to obtain the sensitivity matrix. Default "FDM"
        :type sensitivity_type:str
        :param param_list: A list of float or int of starting values of updating parameters.
        :type param_list: list
        :param target_list: A list of float or int of target/objectives of the updating function handler.
        :type target_list: list
        :param target_resp_list: Multi dimension list of k considered models/cases. If a single dimension list is provided
         the class parses the list into a mutli dimension 1x(k=1) list
        :param max_increm_steps: Maximum updating increments. Default 50
        :param param_increm_rate: Increment rate of parameters between each step. Default 10%
        :param max_error: Error threshold to be minimize.
        """

        # get model and load case inputs
        self.sensitivity_type = kwargs.get(
            "sensitivity_type", "FDM"
        )  # default finite difference method
        self.function_handle = function_handle
        # for output
        self.write_into_txt = kwargs.get("write_output_txt", True)
        self.name = kwargs.get("filename", "Untitled_update_output")
        # init vars of updating
        self.param_list = kwargs.get("param_list", [])  # starting param list
        self.param_bounds_list = kwargs.get(
            "param_bounds_list", [[] for item in self.param_list]
        )  # bounds for params
        # empty list means no bounds.

        # check validity of param bounds
        if len(self.param_list) != len(self.param_bounds_list):
            raise Exception(
                "Param bound list must match the number of parameters: Hint specify empty list [] to params that are not bounded"
            )

        for item in self.param_bounds_list:
            if item and len(item) != 2:
                raise Exception(
                    "For {} bounds should have an upper and lower bound value".format(
                        item
                    )
                )
            if item and item[0] > item[1]:
                raise Exception(
                    "For {} Upper bound has lower value than its lower bound ".format(
                        item
                    )
                )

        self.target_response_list = kwargs.get("target_list", [])  # objectives
        if not isinstance(self.target_response_list[0], list):
            # nest list into a sub list - to accommodate multi model considerations
            self.target_response_list = [self.target_response_list]

        # init var storing history of updating process
        self.sensitivity_matrix = []  # sens matrix local to a single model data set
        self.global_sensitivity_matrix = (
            []
        )  # sens matrix global to all model data set considered
        self.resp_diff = (
            []
        )  # response difference between target ref and current updating step (resets every loop)
        self.global_resp_diff = (
            []
        )  # response difference across all model data set considered
        self.sensitivity_matrix_history = []  # record history of sensitivity matrices
        self.param_update_history = []  # record history of parameter update
        self.response_history = []  # record history of parameter update
        self.resp_diff_history = []  # record history of
        self.residue_history = [0]
        # options of updating
        self.max_increm_steps = kwargs.get("max_increm_steps", 50)
        self.increment_rate = kwargs.get("param_increm_rate", 1.1)
        self.max_error = kwargs.get("max_error", 1)
        self.diagnosis = kwargs.get("diagnosis", False)
        # kwarg to pass into function handler
        self.kwarg = None  # init
        self.update = False

    def set_targets(self, target_list: list):
        """
        Function to set/overwrite target responses - vector R - of updating procedure.
        :param target_list: list of target response value. Note: list order must correspond to the outputs returned
         by main() function. Refer to template main() function for information on setting up the updating procedure.
        :type target_list: list
        """
        self.target_response_list = target_list

    def set_param(self, param_list: list, variance_range=None):
        """
        Function to set/overwrite updating parameters - inputs to main() function.
        :param param_list: list of starting parameter values. Note: list order must correspond to the input of
         main() updating function. Refer to template main() function for information on setting up the updating procedure.
        """
        self.param_list = param_list

    def update_model(self, **kwargs):
        """
        Main function to run updating procedure.

        Function accepts keyword arguments which are then passed to the function handlers of the updating class.

        Example:
            If the function handler takes two keyword arguments -
            function_handler(arg_1=1,arg_2=2)
            ,then the class function is executed as follows -
            update_model(arg_1=1,arg_2=2)
        """

        # set and store initial parameters
        self.param_update_history.append(self.param_list.copy())
        # store kwarg to be passed into function_handle
        self.kwarg = kwargs
        # begin update loop
        # loop through each increment steps
        for j in range(self.max_increm_steps):
            current_param = self.param_update_history[-1]
            # loop for each model considered in the calibration
            for i, ith_target_resp in enumerate(self.target_response_list):
                self._compute_fdm_sensitivity(
                    param_list=current_param, target_resp=ith_target_resp, model_index=i
                )

                self.global_sensitivity_matrix += self.sensitivity_matrix
                self.global_resp_diff += self.resp_diff

            esti = self._get_pseudo_inv_estimation(
                sens_mat=self.global_sensitivity_matrix, resp_diff=self.global_resp_diff
            )
            # compute residue and check threshold

            # update param steps
            param_increments: list = esti.tolist()  # convert np array to list

            param_esti_list = []
            for k, increm in enumerate(param_increments):
                new_kth_param_value = increm + current_param[k]
                if self.param_bounds_list[k]:
                    # bounds specify, check if param are within bounds
                    lb_val = self.param_bounds_list[k][0]
                    ub_val = self.param_bounds_list[k][1]
                    if not lb_val <= new_kth_param_value <= ub_val:
                        # param out of bounds,take the bound values as new value
                        bound_array = np.array([lb_val, ub_val])
                        new_kth_param_value = bound_array[
                            (np.abs(bound_array - new_kth_param_value)).argmin()
                        ]

                param_esti_list.append(new_kth_param_value)

            # param_esti_list = [a + b for (a, b) in zip(param_increments, current_param)]
            self.param_update_history.append(param_esti_list)
            # check updating
            # response error
            e_response = np.sqrt(
                sum([resp**2 for resp in self.global_resp_diff])
            ) / np.sqrt(sum([target**2 for target in self.target_response_list[0]]))
            # parameter estimation error
            error = np.sqrt(
                sum([estimates**2 for estimates in param_increments])
            ) / np.sqrt(sum([current for current in current_param]))
            if error < abs(self.max_error):
                break
            # store for each step
            self.sensitivity_matrix_history.append(self.global_sensitivity_matrix)
            self.resp_diff_history.append(self.resp_diff)

            if self.diagnosis:
                self.print_to(current_step=j, error=e_response)

            # reset vars for next updating step
            self.sensitivity_matrix = []
            self.global_sensitivity_matrix = []
            self.resp_diff = []
            self.global_resp_diff = []

        # finalize parameter
        self.update = True
        # write to txt output
        if self.write_into_txt:
            self._write_output_txt()

        return self.param_update_history, self.response_history

    def _compute_fdm_sensitivity(self, param_list, target_resp=None, model_index=0):
        """
        Function to compute sensitivity matrix based on finite difference sensitivity method. The function runs the
        function handle defined during construction of class with the initial parameters as inputs. Then increments
        the parameter values to obtain an incremental responses. Subsequently, assembles the sensitivity matrix.

        :param param_list: Parameters at current updating step.
        :type param_list:list
        :param model_index: Index of output from function_handle which matches the current evaluating target response.

        :returns array of sensitivity matrix stored in sensitivity_matrix var
        """

        # get base response
        response = self.function_handle(param_list, **self.kwarg)

        if not isinstance(response[0], list):
            # store response as a list - for single var output from function_handle
            response = [response]
        self.response_history.append(response)

        # get base response difference
        for k, ref_resp in enumerate(target_resp):
            # check if target response is list,
            if not hasattr(ref_resp, "__len__"):
                # single scalar, get response diff via subtraction
                self.resp_diff.append(ref_resp - response[model_index][k])
            else:
                # target response is a list/array,
                kth_model_response = response[model_index][k]
                if isinstance(ref_resp, dict):
                    # when dict is provided,
                    # in format with two axis,i.e. x (key) and y (value) axis.
                    # to ensure response and target vectors are same length,
                    # interpolate the measurement to obtain the corresponding value at ordinate of numerical model.
                    time = list(ref_resp.keys())
                    val = list(ref_resp.values())
                    model_time = list(response[model_index][k].keys())

                    if len(ref_resp) > len(response):
                        interp_ref_resp = self._interpolate_measurements(
                            data_x=time, data_y=val, model_x=model_time
                        )
                    else:
                        interp_ref_resp = self._interpolate_measurements(
                            data_x=model_time, data_y=val, model_x=time
                        )
                else:
                    # create an absolute axis between 0 to 1
                    model_abs_axis = np.linspace(0, 1, len(kth_model_response))
                    abs_time_measure = np.linspace(0, 1, len(ref_resp))

                    interp_ref_resp = self._interpolate_measurements(
                        data_x=abs_time_measure, data_y=ref_resp, model_x=model_abs_axis
                    )
                    # plt.plot(model_abs_axis, interp_ref_resp)
                    # plt.plot(model_abs_axis, kth_model_response)
                    # plt.show()
                rmse = self._calculate_rmse(
                    ref_response_list=interp_ref_resp,
                    current_response_list=kth_model_response,
                )

                # append rmse to resp disp
                self.resp_diff.append(rmse)

        # init sensitivity matrix
        self.sensitivity_matrix = [[] for i in range(len(target_resp))]

        # obtain the incremental response by incrementing each parameter by increment rate
        # loop through each param
        for j, param in enumerate(param_list):
            inc_param_list = copy.deepcopy(param_list)
            inc_param_list[j] = param * self.increment_rate
            inc_response = self.function_handle(inc_param_list, **self.kwarg)
            # for each response of current param increment, calculate sensitivity, store to sensitivity matrix
            for i in range(len(target_resp)):
                # checks if output from function handler is a list
                if not isinstance(inc_response[0], list):
                    inc_response = [inc_response]

                # checks if the current response is a list
                if hasattr(target_resp[i], "__len__"):
                    # calculate rmse
                    rmse = self._calculate_rmse(
                        ref_response_list=inc_response[model_index][i],
                        current_response_list=response[model_index][i],
                    )
                    self.sensitivity_matrix[i].append(
                        rmse / (param * abs(self.increment_rate - 1))
                    )
                else:
                    self.sensitivity_matrix[i].append(
                        (inc_response[model_index][i] - response[model_index][i])
                        / (param * abs(self.increment_rate - 1))
                    )

    def get_updated_response(self, param_list):

        # get base response
        return self.function_handle(param_list, **self.kwarg)

    def print_to(self, current_step, error):
        # function to print updating status to
        print("Curent update step {}".format(current_step))
        print(self.param_update_history[-1])
        print("Current error/residue")
        print(error)

    def _write_output_txt(self):
        name = self.name + ".txt"
        with open(name, "w") as file_handle:
            # create py file or overwrite existing
            # writing headers and description at top of file
            file_handle.write("# Output updating analysis for: {}\n".format(name))

            # time
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            file_handle.write("# Constructed on:{}\n".format(dt_string))
            # write imports
            file_handle.write("Parameter history \n")
            for val in self.param_update_history:
                np.savetxt(file_handle, val)

            file_handle.write("Response history\n")
            for val in self.response_history:
                np.savetxt(file_handle, val)

            file_handle.write("Target response\n")
            file_handle.write(str(self.target_response_list) + "\n")

    @staticmethod
    def _get_pseudo_inv_estimation(sens_mat, resp_diff):
        """
        Function to estimate parameters for model updating using Pseudo Inverse Method
        :param sens_mat: sensitivity matrix at current updating step
        :type sens_mat: list of size r by p where r and p are the number of responses and parameters respectively
        """
        sens_mat_transpose = np.transpose(sens_mat)
        resp_num = np.array(sens_mat).shape[0]
        param_num = np.array(sens_mat).shape[1]
        # init var
        estimate = None
        if resp_num == param_num:
            estimate = np.dot(np.linalg.inv(sens_mat), resp_diff)
        elif resp_num > param_num:
            pseudo_inv = np.dot(
                np.linalg.inv(np.dot(sens_mat_transpose, sens_mat)), sens_mat_transpose
            )
            estimate = np.dot(pseudo_inv, resp_diff)
        elif resp_num < param_num:
            pseudo_inv = np.dot(
                sens_mat_transpose,
                np.linalg.inv(
                    np.dot(
                        sens_mat,
                        sens_mat_transpose,
                    )
                ),
            )
            estimate = np.dot(pseudo_inv, resp_diff)

        return estimate

    @staticmethod
    def _calculate_rmse(ref_response_list, current_response_list):
        N = len(ref_response_list)

        return np.sqrt(
            sum(
                [
                    (ref - current) ** 2 / N
                    for (ref, current) in zip(ref_response_list, current_response_list)
                ]
            )
        )

    @staticmethod
    def _get_bayesian_param_estimation(sens_mat, resp_diff, current_param, cp, cr):
        """
        Function to perform parameter estimation using Bayesian approach. Here, a gain matrix G is calculated
        from the provided confidence weighing for parameter and responses.
        :param current_param:
        :param cp:
        :parm cr:
        """
        sens_mat_transpose = np.transpose(sens_mat)
        resp_num = np.array(sens_mat).shape[0]
        param_num = np.array(sens_mat).shape[1]

        # init var
        gain_mat = []
        estimate = None
        if resp_num >= param_num:
            gain_mat = np.dot(
                np.dot(
                    np.linalg.inv(
                        cp + np.dot(np.dot(sens_mat_transpose, cr), sens_mat)
                    ),
                    sens_mat_transpose,
                ),
                cr,
            )

        elif resp_num < param_num:
            cp_inv = np.linalg.inv(cp)
            cr_inv = np.linalg.inv(cr)
            gain_mat = np.dot(
                np.dot(cp_inv, sens_mat_transpose),
                np.linalg.inv(
                    cr_inv + np.dot(np.dot(sens_mat, cp_inv), sens_mat_transpose)
                ),
            )

        else:
            raise Exception("Bayesian parameters error")
        estimate = [
            p + gr for (p, gr) in zip(current_param, np.dot(gain_mat, -resp_diff))
        ]
        return estimate

    def get_updating_outputs(self):
        """
        Extracts information of the updating procedure

        :return:
        """
        if self.update:
            pass
        else:
            pass

    @staticmethod
    def _interpolate_measurements(data_x, data_y, model_x):
        """
        Function to interpolate measurement points on numerical/system model given vectors of measurement (1) time seires
        , and (2) values.

        This function is used when length missmatch between measurement vector and model output vector.

        """
        # takes in a vector of common x axis e.g. time ratio of positional times over total movement time window (0 to 1)

        # create interp function using measurment data - linear interp
        f1 = interp1d(data_x, data_y)

        # apply interp function to ospgrillage data to get respctive positional strain values
        return f1(model_x)
        # return the decimated curve value

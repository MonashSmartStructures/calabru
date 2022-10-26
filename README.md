# Calabru

*calabru* is a python module to calibrate models in python environment. 

## Installation

Install using pip:

```
pip install calabru
```

## Structure of updating

There are three main steps in using *calabru*:

1) Create a structured py function handler that creates and analyses a model in Python Environments.
   The function must be able to pass in updating parameters and return desirable responses from the bespoke model.
   
2) Pass the function handler as an input to *calabru*'s `ModelUpdating` object creation. For example:

``` 
update_object = ModelUpdating(function_handle=my_function_handler, 
                                param_list=starting_param_list,
                                target_list=target_responses_list)
 ```
3) Run the calibration using a single object function.
```
update_object.update_model()
```



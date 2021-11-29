#


## baseBO
[source](https://github.com/wangronin/MIP-EGO\blob\master\mipego/base.py\#L125)
```python 
baseBO(
   search_space: SearchSpace, obj_fun: Callable, parallel_obj_fun: Callable = None,
   eq_fun: Callable = None, ineq_fun: Callable = None, model = None,
   eval_type: str = 'list', DoE_size: int = None, n_point: int = 1,
   acquisition_fun: str = 'EI', acquisition_par: dict = {},
   acquisition_optimization: dict = {}, ftarget: float = None, max_FEs: int = None,
   minimize: bool = True, n_job: int = 1, data_file: str = None, verbose: bool = False,
   random_seed: int = None, logger: str = None
)
```




**Methods:**


### .run
[source](https://github.com/wangronin/MIP-EGO\blob\master\mipego/base.py\#L351)
```python
.run()
```


### .step
[source](https://github.com/wangronin/MIP-EGO\blob\master\mipego/base.py\#L356)
```python
.step()
```


### .ask
[source](https://github.com/wangronin/MIP-EGO\blob\master\mipego/base.py\#L365)
```python
.ask(
   n_point = None
)
```


### .tell
[source](https://github.com/wangronin/MIP-EGO\blob\master\mipego/base.py\#L403)
```python
.tell(
   X, y
)
```

---
Feed an observation back to the surrogate model.


**Args**

* **X** (list) : Places where the objective function has already been evaluated.
    Each suggestion is a dictionary where each key corresponds to a
    parameter being optimized.
* **y** (array) : array-like, shape (n,)
    Corresponding values where objective has been evaluated.


### .create_DoE
[source](https://github.com/wangronin/MIP-EGO\blob\master\mipego/base.py\#L455)
```python
.create_DoE(
   n_point = None
)
```

---
Create a design of experiments using Latin Hypercube Sampling.


**Args**

* **n_point** (int, optional) : Number of points to sample. Defaults to None.


**Returns**

* **Solution**  : Solution object with the design of experiment.


### .post_eval_check
[source](https://github.com/wangronin/MIP-EGO\blob\master\mipego/base.py\#L475)
```python
.post_eval_check(
   X
)
```


### .evaluate
[source](https://github.com/wangronin/MIP-EGO\blob\master\mipego/base.py\#L484)
```python
.evaluate(
   X
)
```

---
Evaluate the candidate points and update evaluation info in the dataframe.


**Args**

* **X** (list) : Samples to evaluate.


**Returns**

* **list**  : evaluations of X.


### .update_model
[source](https://github.com/wangronin/MIP-EGO\blob\master\mipego/base.py\#L504)
```python
.update_model()
```

---
Update the surrogate model.


**Returns**

* **float**  : The R^2 score of the model (not that this is on the training data).


### .arg_max_acquisition
[source](https://github.com/wangronin/MIP-EGO\blob\master\mipego/base.py\#L532)
```python
.arg_max_acquisition(
   n_point = None, return_value = False
)
```

---
Global Optimization of the acqusition function / Infill criterion


**Args**

* **n_point** (int, optional) : Number of points to find. Defaults to None.
* **return_value** (bool, optional) : Return only the points or also the acquisition values. Defaults to False.


**Returns**

* **list**  : Candidate solutions in list with optional infill criterium values.


### .check_stop
[source](https://github.com/wangronin/MIP-EGO\blob\master\mipego/base.py\#L574)
```python
.check_stop()
```


### .save
[source](https://github.com/wangronin/MIP-EGO\blob\master\mipego/base.py\#L584)
```python
.save(
   filename
)
```

---
Save the optimizer and all data for a later restart.


**Args**

* **filename** (string) : Location of the file to save.


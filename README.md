# Project of Data Visualization (COM-480)

By Matoba and Pannatier

| Student's name      | SCIPER |
| ------------------- | ------ |
| Arnaud Pannatier    | 246238 |
| Kyle Michael Matoba | 312224 |

[Milestone 1](#milestone-1) • [Milestone 2](#milestone-2) • [Milestone 3](#milestone-3) 

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![Bokeh](https://img.shields.io/badge/Bokeh-%23F94D5C.svg?style=for-the-badge&logo=Bokeh&logoColor=white)](https://bokeh.org)
[![Hydra](https://img.shields.io/badge/Hydra-%2300B4B6.svg?style=for-the-badge&logo=Hydra&logoColor=white)](https://hydra.cc)

This tool is intended to be run on the user's laptop. Below are the instructions on how to run it locally.
However, we also provide a hosted version at the following link: [https://high-dimentional-explorers.herokuapp.com/main_app](https://high-dimentional-explorers.herokuapp.com/main_app).

you can see a demo of our tool in the [screencast](screencast.mp4)

This readme discusses purely the technical aspects of starting 
the application. The particular ergnomics of the visualization 
tool are documented within the app itself.

_With some slight exceptions, our code and logic is developed 
entirely in Python.[^1] This does not mean that we have no app or 
browser-based front end to interact with it._ Rather, this 
boilerplate is delegated to Bokeh, a tool presented in the 
lecture. This was agreed by the course staff in the conception 
stage.

Our code can be found in the `code/` folder

Python dependencies are in `requirements.txt`, e.g. 
`pip3 install -r requirements.txt`.

In order to be able to run it locally you should update the path in 
`configs/paths` so that it matches your installation.

Then you can run the bokeh server with 
```bash
python3 main_app.py paths=<your-path-file> ++usecase=hcas`
```
Where the usecase can be chosen between `cartpole` and `hcas`

All runtime configuration is handled via the Hydra framework.
To modify parameters most simply in an `argparse`-like syntax
do `python3 main_app.py +foo=bar.baz` or 
`python3 main_app.py ++foo=bar.baz`. 

For example you can change the number of points used in the simulation by overriding 

```bash
python main_app.py paths=<youe-path-file> ++usecase=hcas viz.num_points=1000` 
```
In order for the app to do anything interesting, it needs to be
fed with a pickled PyTorch model (not `state_dict`, for simplicity)
and metadata on its inputs need be configured. We set the correspoding files and configs for our two use cases in the repository.

More information on training a model for a particular application
to automated collision avoidance systems is found in the `acas/`
directory. This is a miniaturized deep learning pipeline that
contains most of the features of a realistic workflow. 
We will submit also a trained network on this problem
to give a self-contained example. 

To train an RL agent to solve the cartpole problem, just run `cartpole/cartpole_policy_gradient_simple.py`. 

[^2] The tornado server that backs Bokeh can occasionally get into a 
bad state. A good way to hard-reset it is (assuming it was started on 
the default 5006 port) `` kill -9  `lsof -i:5006 -t` ``.

[^1]: This is because we use PyTorch. There is no credible 
JavaScript-based deep learning framework.

### Controls
To prevent overwhelming the user with excessive options, we have disabled most of the default controls on the plot.

By default, we have implemented a linked lasso select point, which activates upon clicking. This allows you to select multiple points on the plot simultaneously. If you wish to deselect specific points from the lasso selection, simply press [ESC].

To modify the value of all the sampled points along a particular dimension, you can adjust the corresponding slider. Moving the slider for dimension $i$ will update the value for all sampled points along that dimension. If you wish to revert back to the initial randomly sampled values along this dimension, you can click the "Reset" button.

By utilizing the configuration manager, the user can modify the number of points ($n$) and the sampling scheme from the command line. Pressing the "Resample" button will generate a new set of $n$ points based on the selected sampling scheme.

The rotation plot is connected to the grid plots, enabling seamless interaction between them. The user has the flexibility to switch between our two rotation methods as needed. Additionally, a convenient start/stop button is provided to pause the rotation plot at any interesting point.

## Milestone 1 (7th April, 5pm)

**10% of the final grade**

This is a preliminary milestone to let you set up goals for your final project and assess the feasibility of your ideas.

[PDF is an acceptable submission format](https://edstem.org/eu/courses/94/discussion/29290?comment=52044), thus our milestone \#1 is here: https://github.com/com-480-data-visualization/project-2023-high-dimensional-explorers/blob/master/milestone1.pdf.

## Milestone 2 (7th May, 5pm)

**10% of the final grade**

Our second milestone can be found here: https://github.com/com-480-data-visualization/project-2023-high-dimensional-explorers/blob/master/milestone2.pdf.
The building blocks of the visualization can be found in the folder `m2_building_blocks`.

It uses Python and the bokeh library, you can find the needed libraries in the file `requirements.txt`. 
To run the main prototype you can use the following command:
```shell
$ bokeh serve bokeh_slider_example.py
```
which will create a bokeh server and run simulations on the fly.

The other two components don't need to have a backend server running so you can get the corresponding web pages by running.

```shell
$ python grid.py
```
and 

```shell
$ python model_description.py
```



## Milestone 3 (4th June, 5pm)

More information on the project description but to summarize:

[**Hosted Tool**](https://high-dimentional-explorers.herokuapp.com/main_app) - [**Screencast**](screencast.mp4) -[**Process Book**](process_book.pdf)



# Copyright 2016 Grzegorz Milka grzegorzmilka@gmail.com

import numpy as np
from bokeh.layouts import column, row, widgetbox
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.models.widgets import Slider
from bokeh.models.widgets.buttons import Button
from bokeh.plotting import curdoc, figure
from scipy.stats import norm
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


class Distribution2D:
    '''A 2-dimensional distribution.

    A basic 2-dimensional distribution wrapper with independent distributions
    and scipy-like interface.
    '''
    def __init__(self, x_dist, y_dist):
        self.x_dist = x_dist
        self.y_dist = y_dist

    def pdf(self, x, y):
        return self.x_dist.pdf(x) * self.y_dist.pdf(y)

    def rvs(self, size):
        xs = self.x_dist.rvs(size).reshape((size, 1))
        ys = self.y_dist.rvs(size).reshape((size, 1))
        return np.hstack((xs, ys))


class Population:
    '''A population of given 'size' governed by distribution 'dist'.
    '''
    def __init__(self, dist, size):
        self.pop = dist.rvs(size)

        self.dist = dist
        self.size = size

    def rvs(self):
        return self.pop


class Labeling:
    '''Noisy labeling function'''
    def __init__(self, dist, threshold):
        self.dist = dist
        self.threshold = threshold

    def is_true(self, x, y):
        try:
            return np.array([self.dist.pdf(ix, iy) > self.threshold
                             for ix, iy in zip(x, y)], dtype='bool')
        except TypeError:
            return self.dist.pdf(x, y) > self.threshold


def calculate_roc(hijackers, good_users, labeling):
    '''
    Trains and evalutes an SVM model using the noisy labeling.

    Args:
        hijackers, good_users (Population) - Population instances representing
            the hijackers and good_users.
        labeling (Labeling) - Noisy labeling.
    Returns:
        (fpr, tpr) tuple in sklearn's format representing the ROC curve. The ROC
        curve is based on the accurate labels.
    '''
    X = np.vstack((hijackers.rvs(), good_users.rvs()))
    y = np.array([True] * len(hijackers.rvs()) +
                 [False] * len(good_users.rvs()))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)
    y_train = np.array([labeling.is_true(c[0], c[1]) for c in X_train])

    model = svm.SVC()
    y_score = model.fit(X_train, y_train).decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    return (fpr, tpr)


# Distributions present in the model - cornerstone data source. Their parameters
# may be modified via Bokeh's widgets.
hijackers = Population(Distribution2D(norm(0.2, 0.025), norm(0.5, 0.125)), 100)
good_users = Population(Distribution2D(norm(0.5, 0.125), norm(0.5, 0.125)), 100)
labeling = Labeling(Distribution2D(norm(0.3, 0.06), norm(0.5, 0.125)), 0.5)

# Simulation plot

DOC_WIDTH = 1600

# Parameter widgets
MU_START = -1
MU_END = 1
MU_STEP = 0.05
VAR_START = 0.02
VAR_END = 1.0
VAR_STEP = 0.02
hijacking_x_mu = Slider(title="Hijacking mean X", value=0.2, start=MU_START,
                        end=MU_END, step=MU_STEP)
hijacking_x_var = Slider(title="Hijacking var X", value=0.1, start=VAR_START,
                         end=VAR_END, step=VAR_STEP)
hijacking_y_mu = Slider(title="Hijacking mean Y", value=0.0, start=MU_START,
                        end=MU_END, step=MU_STEP)
hijacking_y_var = Slider(title="Hijacking var Y", value=0.5, start=VAR_START,
                         end=VAR_END, step=VAR_STEP)
hijacking_count = Slider(title="Hijacking size", value=100, start=10, end=500.0,
                         step=10)
good_x_mu = Slider(title="Good user mean X", value=0.0, start=MU_START,
                   end=MU_END, step=MU_STEP)
good_x_var = Slider(title="Good user var X", value=0.5, start=VAR_START,
                    end=VAR_END, step=VAR_STEP)
good_y_mu = Slider(title="Good user mean Y", value=0.0, start=MU_START,
                   end=MU_END, step=MU_STEP)
good_y_var = Slider(title="Good user var Y", value=0.5, start=VAR_START,
                    end=VAR_END, step=VAR_STEP)
good_count = Slider(title="Good user size", value=300, start=10, end=500.0,
                    step=10)
labeling_x_mu = Slider(title="Labeling mean X", value=0.25, start=MU_START,
                       end=MU_END, step=MU_STEP)
labeling_x_var = Slider(title="Labeling var X", value=0.2, start=VAR_START,
                        end=VAR_END, step=VAR_STEP)
labeling_y_mu = Slider(title="Labeling mean Y", value=0.05, start=MU_START,
                       end=MU_END, step=MU_STEP)
labeling_y_var = Slider(title="Labeling var Y", value=0.4, start=VAR_START,
                        end=VAR_END, step=VAR_STEP)
labeling_thr = Slider(title="Labeling threshold", value=0.5, start=0, end=2.0,
                      step=0.05)


# Bokeh's data source for the simulation plot and controller functions.
data_plot_text_source = ColumnDataSource(data=dict(text=['']))
data_plot_hcircle_source = ColumnDataSource(data=dict(x=[], y=[], line=None))
data_plot_gcircle_source = ColumnDataSource(data=dict(x=[], y=[], color=[],
                                                      line=None))
data_plot_back_source = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[],
                                                   dh=[]))


LABEL_COLOR = 'black'
H_COLOR = 'red'
G_COLOR = 'green'
RADIUS = 0.014


def get_labels(p, labeling):
    labels = labeling.is_true(p[:, 0], p[:, 1])
    return labels


def update_avr_text():
    global data_plot_gcircle_source
    global hijackers
    global good_users
    global labeling
    global data_plot_text_source
    h = hijackers.rvs()
    g = good_users.rvs()
    hl = np.sum(get_labels(h, labeling))
    gl = np.sum(get_labels(g, labeling))
    data_plot_text_source.data = dict(
        text=['Accuracy: {0:.2f}, Recall: {1:.2f}'.format(hl / (hl + gl),
                                                          hl / len(h))])


def update_circles():
    global hijackers
    global good_users
    global labeling
    global data_plot_hcircle_source
    global data_plot_gcircle_source
    h = hijackers.rvs()
    g = good_users.rvs()
    hlines = [LABEL_COLOR if l else H_COLOR for l in get_labels(h, labeling)]
    glines = [LABEL_COLOR if l else G_COLOR for l in get_labels(g, labeling)]
    data_plot_hcircle_source.data.update({'x': h[:, 0], 'y': h[:, 1], 'line':
                                          hlines})
    data_plot_gcircle_source.data.update({'x': g[:, 0], 'y': g[:, 1], 'color':
                                          [G_COLOR] * good_count.value, 'line':
                                          glines})


def update_hijackers():
    global hijackers
    hijackers = Population(Distribution2D(norm(hijacking_x_mu.value,
                                               hijacking_x_var.value),
                                          norm(hijacking_y_mu.value,
                                               hijacking_y_var.value)),
                           hijacking_count.value)


def update_good_users():
    global good_users
    good_users = Population(Distribution2D(norm(good_x_mu.value,
                                                good_x_var.value),
                                           norm(good_y_mu.value,
                                                good_y_var.value)),
                            good_count.value)


def update_threshold_source():
    global labeling
    labeling = Labeling(Distribution2D(norm(labeling_x_mu.value,
                                            labeling_x_var.value),
                                       norm(labeling_y_mu.value,
                                            labeling_y_var.value)),
                        labeling_thr.value)


def update_back_source():
    global labeling
    labeling = Labeling(Distribution2D(norm(labeling_x_mu.value,
                                            labeling_x_var.value),
                                       norm(labeling_y_mu.value,
                                            labeling_y_var.value)),
                        labeling_thr.value)
    l = labeling
    N = 40
    W = MU_END - MU_START
    H = W
    img = np.empty((N, N), dtype=np.uint32)
    pdfs = np.empty((N, N), dtype=np.float64)
    view = img.view(dtype=np.uint8).reshape((N, N, 4))
    half_square = 1 / (2 * N)
    for i in range(N):
        for j in range(N):
            view[i, j, 0] = 255
            view[i, j, 1] = 71
            view[i, j, 2] = 71
            pdfs[i, j] = l.dist.pdf(MU_START + W * (j / N + half_square),
                                    MU_START + H * (i / N + half_square))

    pdf_max = pdfs.max()
    for i in range(N):
        for j in range(N):
            view[i, j, 3] = int(128 * pdfs[i, j] / pdf_max)
    data_plot_back_source.data = dict(image=[img], x=[MU_START], y=[MU_START],
                                      dw=[W], dh=[H])

update_hijackers()
update_good_users()
update_back_source()
update_circles()
update_avr_text()


# Data sources are now prepared, draw the initial plot and wire up controllers
# to the widgets.
space_plot = figure(title='Training space', x_axis_label='risky feature',
                    y_axis_label='independent feature', x_range=(MU_START,
                                                                 MU_END),
                    y_range=(MU_START, MU_END), width=1000)
space_plot.image_rgba(image="image", source=data_plot_back_source, x='x', y='y',
                      dw='dw', dh='dh')
space_plot.circle(x="x", y="y", source=data_plot_hcircle_source, radius=RADIUS,
                  fill_color=H_COLOR, line_color='line', line_width=2.5,
                  fill_alpha=0.7)
space_plot.circle(x="x", y="y", source=data_plot_gcircle_source, radius=RADIUS,
                  fill_color=G_COLOR, line_color='line', line_width=2.5,
                  fill_alpha=0.7)
space_plot.circle(x=-100, y=-100, radius=RADIUS, fill_color=H_COLOR,
                  line_color='black', line_width=2.5, fill_alpha=1.0,
                  legend="Labeled hijacker")
space_plot.circle(x=-100, y=-100, radius=RADIUS, fill_color=H_COLOR,
                  line_color=H_COLOR, line_width=2.5, fill_alpha=1.0,
                  legend="Unlabeled hijacker")
space_plot.circle(x=-100, y=-100, radius=RADIUS, fill_color=G_COLOR,
                  line_color=LABEL_COLOR, line_width=2.5, fill_alpha=1.0,
                  legend="Labeled good user")
space_plot.circle(x=-100, y=-100, radius=RADIUS, fill_color=G_COLOR,
                  line_color=G_COLOR, line_width=2.5, fill_alpha=1.0,
                  legend="Unlabeled good user")
acc_vs_rec_label = LabelSet(x=70, y=70, x_units='screen', y_units='screen',
                            text='text', render_mode='css',
                            source=data_plot_text_source,
                            border_line_color='black', border_line_alpha=1.0,
                            background_fill_color='white',
                            background_fill_alpha=1.0)
space_plot.add_layout(acc_vs_rec_label)


def hijacking_callback():
    update_hijackers()
    update_circles()
    update_avr_text()


def good_users_callback():
    update_good_users()
    update_circles()
    update_avr_text()


def labeling_callback():
    update_back_source()
    update_circles()
    update_avr_text()


for widget in [hijacking_x_mu, hijacking_x_var, hijacking_y_mu, hijacking_y_var,
               hijacking_count]:
    widget.on_change('value', lambda attr, old, new: hijacking_callback())

for widget in [good_x_mu, good_x_var, good_y_mu, good_y_var, good_count]:
    widget.on_change('value', lambda attr, old, new: good_users_callback())

for widget in [labeling_x_mu, labeling_x_var, labeling_y_mu, labeling_y_var,
               labeling_thr]:
    widget.on_change('value', lambda attr, old, new: labeling_callback())

hijacking_input = widgetbox(hijacking_x_mu, hijacking_y_mu, hijacking_x_var,
                            hijacking_y_var, hijacking_count)
good_user_input = widgetbox(good_x_mu, good_y_mu, good_x_var,
                            good_y_var, good_count)
labeling_input = widgetbox(labeling_x_mu, labeling_y_mu, labeling_x_var,
                           labeling_y_var, labeling_thr)
inputs = [hijacking_input, good_user_input, labeling_input]

# Plot the ROC curve

roc_source = ColumnDataSource(data=dict(x=[], y=[]))


def train_and_evaluate():
    global hijackers, good_users, labeling, roc_source
    fpr, tpr = calculate_roc(hijackers, good_users, labeling)
    roc_source.data = dict(x=fpr, y=tpr)


train_and_evaluate()
roc = figure(title='RoC', x_axis_label='False positive rate',
             y_axis_label='True Positive Rate', x_range=(0, 1), y_range=(0, 1),
             width=1000)
roc.line([0, 1], [0, 1], color='navy', line_width=1.0, line_dash='dashed')
roc.line(x='x', y='y', source=roc_source, color='darkorange', line_width=1.0,
         line_dash='dashed')

button = Button(label='Calculate ROC')
button.on_click(train_and_evaluate)

# Add the simulation and ROC curve to DOM.

curdoc().add_root(column(row(space_plot, *inputs), row(roc, widgetbox(button))))
curdoc().title = 'Recall vs accuracy trade-off'

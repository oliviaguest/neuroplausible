---

published: true
title: Artificial Neural Networks with Random Weights are Baseline Models
tldr:
  repos: [random-network, brain-imaging-and-the-neural-code]
  pubs: [guest_love_2017]
  info: Running untrained networks with random weights allows us to understand such models before training.

img: /img/posts/ann_models_correlation
permalink: random-network
author: Olivia Guest

---

Where do the impressive performance gains of deep neural networks come from?
Is their power due to the learning rules which adjust the connection weights or is it simply a function of the network architecture (i.e., many layers)?
These two properties of networks are hard to disentangle.
One way to tease apart the contributions of network architecture versus those of the learning regimen is to consider networks with randomised weights.
To the extent that random networks show interesting behaviors, we can infer that the learning rule has not played a role in them.
At the same time, examining these random networks allows us to evaluate what learning does add to the network's abilities over and above minimising some loss function.

<div class="float-right figure">
  <object class="image" data="{{ site.baseurl}}{{ page.img }}.svg" type="image/svg+xml">
    <img src="{{ site.baseurl}}{{ page.img }}.png" />
  </object>
  <div class="figure-caption">
    <a href="https://elifesciences.org/content/6/e21397#fig2">Figure 2A</a> from Guest and Love (2017): "For the artificial neural network coding schemes, similarity to the prototype falls off with increasing distortion (i.e., noise). The models, numbered 1–11, are (<i>1</i>) vector space coding, (<i>2</i>) gain control coding, (<i>3</i>) matrix multiplication coding, (<i>4</i>), perceptron coding, (<i>5</i>) 2-layer network, (<i>6</i>) 3-layer network, (<i>7</i>) 4-layer network, (<i>8</i>) 5-layer network, (<i>9</i>) 6-layer network (<i>10</i>) 7-layer network, and (<i>11</i>), 8-layer network. The darker a model is, the simpler the model is and the more the model preserves similarity structure under fMRI."
  </div>  
</div>

In _[What the Success of Brain Imaging Implies about the Neural Code](http://dx.doi.org/10.7554/eLife.21397)_, we examined an artificial deep neural network, Inception-v3 GoogLeNet.
This deep trained network, preserves the similarity of the input space and thus is [functionally smooth](https://elifesciences.org/content/6/e21397#s2).
Importantly, however, we found that functional smoothness in this deep network breaks down at later layers.
Is this because of the depth of the network, the many layers, or the specific learning regimen?
We sought to explain why this happens by using a baseline, a model with random weights.

To answer this question, let us consider some much simpler plausible contenders for the neural code — a rudimentary set of models — the components of artificial neural networks: [matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication) and some kind of squashing ([sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function), [step](https://en.wikipedia.org/wiki/Step_function), [etc](https://en.wikipedia.org/wiki/Activation_function).) function (in our case, the [hyperbolic tangent](https://en.wikipedia.org/wiki/Hyperbolic_function#Hyperbolic_tangent)).


The first basic model, matrix multiplication, is how neural networks propagate activation from layer $$\mathbf{m}$$ to the next $$\mathbf{n}$$ via the weights $$\mathbf{w}$$.
For simplicity, our toy network contains layers $$\mathbf{m}$$ and $$\mathbf{n}$$, which both contain three units.
Thus to calculate the states for $$\mathbf{n}$$, we take the matrix product of the previous layer $$\mathbf{m}$$ and the weights $$\mathbf{w}$$:

$$

\mathbf{m} \times \mathbf{w}
=
\\
\begin{pmatrix}
x_1 & x_2 & x_3 \\
\end{pmatrix}
\times
\begin{pmatrix}
w_{11} & w_{12} & w_{13} \\
w_{21} & w_{22} & w_{23} \\
w_{31} & w_{32} & w_{33}
\end{pmatrix}
=
\\
\begin{pmatrix}
x_1 w_{11} + x_2 w_{21} + x_3 w_{31} \\
x_1 w_{12} + x_2 w_{22} + x_3 w_{32} \\
x_1 w_{13} + x_2 w_{23} + x_3 w_{33}
\end{pmatrix}\
=
\begin{pmatrix}
y_1 & y_2 & y_3
\end{pmatrix}
=
\mathbf{n}
\,
$$

where $$x$$s represent the units in layer $$\mathbf{m}$$, $$w_{ij}$$ represents a weight in $$\mathbf{w}$$ from unit $$i$$ in layer $$\mathbf{m}$$ to unit $$j$$ in $$\mathbf{n}$$, and $$y_j$$ is a unit in $$\mathbf{n}$$. For example, $$w_{31}$$ is the weight on the connection between the third unit of the shallower/earlier layer and the first unit of the deeper/later later (others use other notations).

Matrix multiplication calculates the states of a layer — easily done in Python using [NumPy](http://www.numpy.org/), specifically [```numpy.dot()```](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html):

{% highlight python  %}
import numpy as np
m  = np.asarray([0.1, 0.2, 1.3]) # layer m with some dummy input
w = np.random.randn(3, 3) # random weights from m to n
n = np.dot(m, w) # pre-synaptic states in n
print(n)
{% endhighlight %}

To apply a squashing function, $$\tanh$$, to ```n``` above, we may use [```numpy.tanh()```](https://docs.scipy.org/doc/numpy/reference/generated/numpy.tanh.html):
{% highlight python  %}
n = np.tanh(n) # post-synaptic states in n
print(n)
{% endhighlight %}
Non-linear transformations like hyperbolic tangent allow the network to have non-linear decision boundaries, e.g., between classes, making it able to capturing the statistics of the training set (more [here](http://www.kdnuggets.com/2016/08/role-activation-function-neural-network.htmlhttp://www.kdnuggets.com/2016/08/role-activation-function-neural-network.html) and [here](https://www.quora.com/Why-do-neural-networks-need-an-activation-function/answer/Chomba-Bupe)).

In [Guest and Love (2017)](http://dx.doi.org/10.7554/eLife.21397) we presented the above as two separate models as well as a combined model, here I have cut to the part where they are combined to form a traditional two-layer network (also known as the [perceptron](https://en.wikipedia.org/wiki/Perceptron) model).
As you might have guessed, from two layers we can generalise to many, by continuing to take the matrix product of the output (```n``` in the code above) with some new weights, and so on.

Running an untrained neural network with random weights allows us to compare more complex (i.e., trained models) with their untrained selves.
We can thus pick apart what aspects of the model are inherent to the architecture itself and which emerge as a function of training.
Networks that have random weights can be given the same training and test sets, although importantly no training has happened yet, and we can examine their internal states and outputs
This can serve as a guide to understand what the network "knows" a priori.

As we noted in [Guest and Love (2017)](http://dx.doi.org/10.7554/eLife.21397), networks naturally place items close together in their internal representational space that are similar/proximal in the input space. Hence why artificial neural networks are a plausible candidate for the neural code, i.e., they give rise to [functionally smooth](https://elifesciences.org/content/6/e21397#s2) representations.
The simple network above can be made deeper and deeper, and we can inspect every layer in it for smoothness for every pattern.
Extending the above, we can do just that, and run the network on two very simple categories:

{% highlight python linenos=table %}
import numpy as np

prototypes  = np.random.randn(2, 100) # two toy categories
members = 10 # how many items per category
patterns = []

for p in prototypes:
    for i, m in enumerate(range(members)):
        # for each item, create a pattern that has noise as a function of the
        # number of items. First item in category has no noise, then 0.05 SD of
        # noise, then 0.1 SD, and so on.
        patterns.append(p + 0.01 * i * np.random.randn((len(p))))

layers = 20 # how many layers we want, i.e., how deep is the network
# random weights:
w = np.random.randn(layers, len(prototypes[0]), len(prototypes[0])) * 0.1

for pat in patterns:
    # for each pattern
    for i, l in enumerate(range(layers)):
        if i == 0:
            #if we are at the input layer, then set units to pattern
            n = pat
        # propagate through each layer
        n = np.dot(n, w[i]) # pre-synaptic states in n
        n = np.tanh(n) # post-synaptic states in n
        if i == layers-1:
            # print the layer, the first five features of the pattern applied at
            # input and the first five activations in the last layer
            print i, pat[0:5], n[0:5]
{% endhighlight %}
Even just by eye-balling the output in the terminal using [the code above](https://github.com/oliviaguest/random-network), we can see that indeed similar items (items within the same category) map to similar outputs, i.e., the network is functionally smooth without any training. We used a [more complex version of the above](https://github.com/oliviaguest/brain-imaging-and-the-neural-code/tree/master/random-network) to demonstrate this principle in [Guest and Love (2017)](http://dx.doi.org/10.7554/eLife.21397), where we calculate the correlations between the representations in the input space and in each layer.
However, as we move deeper into the network, we see that functional smoothness has broken down and the network gives for all intents and purposes identical outputs for each items within a category, thus losing all structure within it.
We cannot looking just at the output, predict which input generated it, only which category.

Using this result we can infer that the property of Inception-v3 GoogLeNet, and indeed any similar deep network, which causes it to both display (at early layers) and gradually lose functional smoothness (at deeper layers), is due to the nature of the architecture and not the learning rule.
Because this property is present in simple untrained networks, it cannot be a byproduct of training.

Importantly, randomising weights can be done to any network with any topology, including to Inception-v3 GoogLeNet itself, to recurrent networks, and so on.
We hope this idea proves to be a useful exercise to others too, as many connectionist and deep network accounts would benefit from an understanding of the inherent properties of the topological configuration versus the fully-trained model.

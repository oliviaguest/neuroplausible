---

published: true
title: "Block Practical: Connectionist Models and Cognitive Processes"
tldr:
  repos: connectionism
  info: These are the course materials I used to teach second year undergraduate students basic programming in Python and connectionism in the Experimental Psychology Department, University of Oxford.

img:
  svg: true
  dir: /img/posts/neural_network
permalink: connectionism
author: Olivia Guest

---

This is less of a blog post and more of a materials dump from an elective practical I taught to second year undergraduate students in the Experimental Psychology Department at the University of Oxford. I thoughtlessly deleted the webpage that contained them, assuming no student after 2 years would need them. How wrong I was! I received an email the other day from a Ph.D. student at a university on the other side of the world pretty much asking where these materials had disappeared to. This made me question my assumption nobody was looking at these materials. So to save myself and others from looking for them again, here they are for everybody.

This elective practical taught second year undergraduates to program in Python at a basic level and to understand the basics of artificial neural networks. They proved highly suitable as my students had not done much/any programming before and had not really heard of neural networks (things might have changed now, hype, etc).

To clarify, I do not teach this course any more and I will not be updating or using these materials. If you want to use them for your own teaching, they are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and I would super appreciate an <a href="mailto:o.guest@ucl.ac.uk">email</a> or <a href="https://twitter.com/o_guest">tweet me</a> if you use them.

<div class="float-right figure">
  <object class="image" data="{{ site.baseurl}}{{ page.img.dir }}.svg" type="image/svg+xml">
    <img src="{{ site.baseurl}}{{ page.img.dir }}.png" />
  </object>
  <div class="figure-caption">
    A really basic neural network diagram, by Wikipedia user <a href="https://commons.wikimedia.org/wiki/File:Colored_neural_network.svg">Glosser.ca</a>
  </div>  
</div>




## Course Materials
### 1st Week: Introduction to Programming and Connectionist Networks
- Code: [pyceptron.py](https://github.com/oliviaguest/connectionism/raw/master/week1/pyceptron.py)
- Slides: [Part 1: Intro to Programming](https://github.com/oliviaguest/connectionism/raw/master/week1/slides/part_1_slides.pdf), [Part 2: Intro to Networks](https://github.com/oliviaguest/connectionism/raw/master/week1/slides/part_2_slides.pdf)
- Exercises: [Pyceptron](https://github.com/oliviaguest/connectionism/raw/master/week1/exercises/exercises.pdf)

### 2nd Week: Going from Two Network Layers to Three
- Code: [network_missing.py](https://github.com/oliviaguest/connectionism/raw/master/week2/network_missing.py), [network_hints.py](https://github.com/oliviaguest/connectionism/raw/master/week2/network_hints.py), [network.py](https://github.com/oliviaguest/connectionism/raw/master/week2/network.py)
- Slides: [Part 3: Feedfoward Networks](https://github.com/oliviaguest/connectionism/raw/master/week2/slides/part_3_slides.pdf)
- Exercises: [Backpropagation](https://github.com/oliviaguest/connectionism/raw/master/week2/exercises/exercises.pdf)

### 3rd Week: Replicating a Model
- Code: [network.py](https://github.com/oliviaguest/connectionism/raw/master/week3/network.py)
- Patterns: [tyler_patterns.csv](https://github.com/oliviaguest/connectionism/raw/master/week3/tyler_patterns.csv)
- Slides: [Part 4: Replicating a Model](https://github.com/oliviaguest/connectionism/raw/master/week3/slides/part_4_slides.pdf)
- Exercises: [Replication of Tyler et al. (2000)](https://github.com/oliviaguest/connectionism/raw/master/week3/exercises/exercises.pdf)
- Tyler, L. K., Moss, H. E., Durrant-Peatfield, M. R., & Levy, J. P. (2000). **[Conceptual structure and the structure of concepts: A distributed account of category-specific deficits](https://github.com/oliviaguest/connectionism/raw/master/week3/tyler_2000.pdf)**. *Brain and Language*, 75(2), 195-231.

### 4th Week: Writing up Experimental Results
- Code: [network.py](https://github.com/oliviaguest/connectionism/raw/master/week4/network.py), [graph.py](https://github.com/oliviaguest/connectionism/raw/master/week4/graph.py)
- Example file for errors: [errors1000.txt](https://github.com/oliviaguest/connectionism/raw/master/week4/errors1000.txt)
- Slides: [Part 5: Writing the Report](https://github.com/oliviaguest/connectionism/raw/master/week4/slides/part_5_slides.pdf)
- Exercises: [File Input/Output](https://github.com/oliviaguest/connectionism/raw/master/week4/exercises/exercises.pdf)

## Reading Materials
- [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528v2)
- [Essay: A Brief Introduction to Connectionism](http://kimplunkett.org.uk/secondtry/page31/page32/index.html)

## Programming
### Exercises
- [Codecademy](www.codecademy.com)
- [LearnPython.org](http://www.learnpython.org/)
- [Codewars](http://www.codewars.com/)
- [Code School: Python](https://www.codeschool.com/paths/python)

### Books
- [Learn Python the Hard Way](http://learnpythonthehardway.org/book/)
- [How to Think Like a Computer Scientist: Learning with Python](http://www.openbookproject.net/thinkcs/python/english2e/)
- [Think Python: How to Think Like a Computer Scientist](http://www.greenteapress.com/thinkpython/)

## Inspiration
### Libraries
- [Numpy Tutorial](http://www.python-course.eu/numpy.php)
- [Matplotlib Examples](http://matplotlib.org/1.4.0/examples/index.html)
- [A Primer on Scientific Programming with Python](https://hplgit.github.io/scipro-primer/slides/index.html)
- [Scipy Lecture Notes](http://www.scipy-lectures.org)

### Blogs
- [The Glowing Python](http://glowingpython.blogspot.co.uk/): This blog has various examples of interesting code to play with and give you ideas for your own projects.
- WildML: [Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/): This blog also has other Machine Learning tutorials.

### Video Lectures
- [Machine Learning](https://www.youtube.com/playlist?list=PLg7f-TkW11iX3JlGjgbM2s8E1jKSXUTsG), by The Royal Society
- [The Cognitive and Computational Neuroscience of Categorization, Novelty-Detection, and the Neural Representation of Similarity](https://www.youtube.com/watch?v=2Ei6wFJ9kCc), by Mark Gluck

## Online Courses
- [Machine Learning](https://www.coursera.org/learn/machine-learning/), by Andrew Ng
- [Neural Networks for Machine Learning](https://www.coursera.org/course/neuralnets), by Geoffrey Hinton
- [Introduction to Neural Networks](http://ocw.mit.edu/courses/brain-and-cognitive-sciences/9-641j-introduction-to-neural-networks-spring-2005/index.htm), by Sebastian Seung

## How to install Python
### Windows Users
This is a little tricky:
1. Install Python: [download from here](https://www.python.org/ftp/python/2.7.10/python-2.7.10.msi)

2. Install matplotlib, numpy, and scipy using pip. Specifically you need to download the following from [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/):
 - matplotlib-1.4.3-cp27-none-win32.whl
 - numpy-1.10.0b1+mkl-cp27-none-win32.whl
 - scipy-0.16.0-cp27-none-win32.whl

 This requires you to be in the Scripts folder of the Python27 installation. And to use the windows command prompt. For me this looks like:
```
C:\Python27\Scripts>pip install NAME_OF_WHEEL_FILE.whl
```
For all three of those you need to run a pip command like above.  

3. Install PyGTK: [download from here](http://ftp.gnome.org/pub/GNOME/binaries/win32/pygtk/2.24/pygtk-all-in-one-2.24.2.win32-py2.7.msi)

4. To check that everything works, open network.py and see if it runs without any errors.

### Mac Users
I finally managed to do this on my mac. Use [Homebrew](http://brew.sh/) to install matplotlib, numpy, scipy, pygtk.

### Linux Users
Use your favourite package manager to install matplotlib, numpy, scipy, pygtk.

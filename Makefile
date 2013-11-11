PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests

# Compilation...

CYTHONSRC= $(wildcard mrec/*/*.pyx)
CSRC= $(CYTHONSRC:.pyx=.cpp)

inplace: cython
	$(PYTHON) setup.py build_ext -i

cython: $(CSRC)

clean:
	rm -f mrec/*/*.c mrec/*/*.so mrec/*/*.html mrec/*/*.pyc

%.cpp: %.pyx
	$(CYTHON) $<

# Tests...
#
test-code:
	$(NOSETESTS) -s mrec

test-coverage:
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=mrec mrec

test: test-code


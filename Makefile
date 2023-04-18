.PHONY: all clean
all: train test

clean:
	@echo "********** CleanUp **********"
	@echo "In developing process"


train:
	@echo "********** Training **********"
	python train.py

test:
	@echo "********** Testing **********"
	@echo "In developing process"

tensorboard:
    @echo "******* Tensorboard *********"
    tensorboard --logdir=runs

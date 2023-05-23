CC 		= gcc
INC 	= ../include/
CFLAGS	= -Wall -Wextra -Werror -I $(INC) -g
SRC 	= ../src/
EXEC	= mnist_train.x mnist_test.x mnist_cnn_train.x

all		: $(EXEC)

mnist_train.x : mnist_train.o $(SRC)network.o $(SRC)ndarray.o $(SRC)layer.o $(SRC)misc.o
	$(CC) $(CFLAGS) $^ -o $@ -lm

mnist_test.x : mnist_test.o $(SRC)network.o $(SRC)ndarray.o $(SRC)layer.o $(SRC)misc.o
	$(CC) $(CFLAGS) $^ -o $@ -lm

mnist_cnn_train.x : mnist_cnn_train.o $(SRC)cnn.o $(SRC)ndarray.o $(SRC)layer.o $(SRC)misc.o
	$(CC) $(CFLAGS) $^ -o $@ -lm
	
$(SRC)%.o	: $(SRC)%.c
	$(CC) $(CFLAGS) -c $< -o $@ -lm

.PHONY : clean realclean
clean:
	rm -rf $(SRC)*.o *.o

realclean: clean
	rm -rf *.x
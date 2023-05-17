CC 		= gcc
INC 	= ../include/
CFLAGS	= -Wall -Wextra -Werror -I $(INC) -g
SRC 	= ../src/
EXEC	= test_ndarray.x test_network.x

all		: $(EXEC)

test_ndarray.x : test_ndarray.o $(SRC)ndarray.o
	$(CC) $(CFLAGS) $^ -o $@ -lm

test_network.x : test_network.o $(SRC)network.o $(SRC)ndarray.o $(SRC)layer.o
	$(CC) $(CFLAGS) $^ -o $@ -lm
	
$(SRC)%.o	: $(SRC)%.c
	$(CC) $(CFLAGS) -c $< -o $@ -lm

.PHONY : clean realclean
clean:
	rm -rf $(SRC)*.o *.o

realclean: clean
	rm -rf *.x

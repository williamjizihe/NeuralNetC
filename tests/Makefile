CC 		= gcc
INC 	= ../include/
CFLAGS	= -Wall -Wextra -Werror -I $(INC) -g
SRC 	= ../src/
EXEC	= test_matrix.x

all		: $(EXEC)

test_matrix.x : test_matrix.o $(SRC)matrix.o
	$(CC) $(CFLAGS) $^ -o $@

$(SRC)%.o	: $(SRC)%.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY : clean realclean
clean:
	rm -rf $(SRC)*.o *.o

realclean: clean
	rm -rf *.x

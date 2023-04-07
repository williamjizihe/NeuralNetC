CC 		= gcc
INC 	= include/
CFLAGS	= -Wall -Wextra -Werror -I $(INC) -g
SRC 	= src/
EXEC	= examples/main.x

all		: $(EXEC)

examples/main.x : $(SRC)main.o $(SRC)*.o
	$(CC) $(CFLAGS) $^ -o $@

$(SRC)%.o	: $(SRC)%.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY : clean realclean
clean:
	rm -rf $(SRC)*.o

realclean: clean
	rm -rf $(SRC)*.x

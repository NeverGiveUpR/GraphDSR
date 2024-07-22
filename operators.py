class Operators:
    arity_dict = {
        "add": 2, 
        "div": 2, 
        "sub": 2, 
        "mul": 2,
        "sin": 1, 
        "cos": 1, 
        "sqrt": 1, 
        "square": 1, 
        "exp": 1, 
        "log": 1,
        "x1": 0,
        "x2": 0,
        "x3": 0,
        "c": 0,
        "P": 0
    }

    def __init__(self, operator_list, var_number):
        self.library = []
        if "c" in operator_list:
            operator_list.remove("c")
            self.library = ['c']
        self.operator_list = operator_list
        self.operator_num = len(self.operator_list)
        self.var_num = var_number
        self.var_list = ['x'+str(i+1) for i in range(self.var_num)]
        self.library = self.operator_list+self.var_list+self.library+['P']
        # print("self.library:", self.library)
        self.var_list_index = [i for i in range(len(self.library)) if self.library[i] in self.var_list]
        self.trig_index = [i for i in range(len(self.library)) if self.library[i] in ['sin', 'cos', 'tanh']]

        # allocate token for each operator
        self.token2id = {}
        self.id2token = {}
        self.library_lst = []
        for i in range(len(self.library)):
            self.token2id[self.library[i]] = i
            self.id2token[i] = self.library[i]
            self.library_lst.append(i)

        print("self.token2id:", self.token2id)
        print("self.id2token:", self.id2token)

    def arity_i(self, index):
        return self.arity_dict[self.id2token[int(index)]]

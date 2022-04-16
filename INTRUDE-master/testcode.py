outfile = 'ccc.txt'


with open("data/clf/second_nondup.txt") as f:
    arr = f.readlines()
    arr = [i.replace("\n","") for i in arr]
    for value in arr:
        if value.startswith("elas"):
            with open("ccc.txt", 'a+') as outf:
                outf.write(value+"\n")
                pass
            pass
        pass
    pass




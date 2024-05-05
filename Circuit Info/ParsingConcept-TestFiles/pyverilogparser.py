import pyverilog.vparser.parser as prs


with open('c3540.v','r') as f:
    vc = f.read() 


# # a = prs.VerilogCodeParser(vc).parse()
# # a = prs.VerilogParser().parse(text=vc)


a , b = prs.parse(vc)

small = a.show()
print("--------------")



# for lineno, directive in b:
#     print("Line %d : %s"%(lineno, directive))




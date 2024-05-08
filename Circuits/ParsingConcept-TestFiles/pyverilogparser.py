import pyverilog.vparser.parser as prs


with open('vga_ctrl.v','r') as f:
    vc = f.read() 


# # a = prs.VerilogCodeParser(vc).parse()
# # a = prs.VerilogParser().parse(text=vc)


a , b = prs.parse(vc)
# a.show()

small = a.children()
print(a.show())
print("--------------")



# for lineno, directive in b:
#     print("Line %d : %s"%(lineno, directive))




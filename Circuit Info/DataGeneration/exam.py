import pyverilog.vparser.parser as parses



with open('c17.v','r') as f:
    vc = f.read()

a,b = parses.parse(vc)



a.show()
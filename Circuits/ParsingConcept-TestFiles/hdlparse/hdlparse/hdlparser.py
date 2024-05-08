import verilog_parser as vlog


with open('c3540.v','r') as f:
  vc = f.read()

vlog_ex = vlog.VerilogExtractor()
vlog_mods = vlog_ex.extract_objects_from_source(vc)


print("-----------------------------------")
# print(vlog_mods)
print("-----------------------------------")
for m in vlog_mods:
  print('Module "{}":'.format(m.name))

  print('  Parameters:')
  for p in m.generics:
    print('\t{:20}{:8}{}'.format(p.name, p.mode, p.data_type))

  print('  Ports:')
  for p in m.ports:
    print('\t{:20}{:8}{}'.format(p.name, p.mode, p.data_type))
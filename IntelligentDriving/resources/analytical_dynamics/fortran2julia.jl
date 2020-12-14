function main(fname)
  fp = open(fname, "r")
  data = read(fp, String)
  close(fp)

  global_rules = [
                  r"\s*\n\s+[0-9]\s+" => s"", 
                 ]
  for rule in global_rules
    data = replace(data, rule)
  end

  line_rules = [
           r"\s+" => s"", 
           r"\*\*" => s"^",
           r"([A-Za-z][a-zA-Z0-9_]*)\(([0-9,]+)\)" => s"\1[\2]",
           r"([0-9.]+)[dDeE]([-+][0-9]+)" => s"\1e\2",
          ]
  lines = split(data, '\n')
  for rule in line_rules
    lines = map(line -> replace(line, rule), lines)
  end
  return join(lines, '\n')
end

data = main(realpath(ARGS[1]))
print(data)

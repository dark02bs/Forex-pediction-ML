require 'torch'
require 'nn'
require 'optim'
require 'gnuplot'
require 'paths'

create_data = function(filename)
	input = {}
	output = {}
	i=0
	for line in io.lines(filename) do
		i=i+1
		st_pr,opn,cls,ans=unpack(line:split(";"))
		
		local t={st_pr,opn,cls}
		table.insert(input,t)
		table.insert(output,{ans})
		if i==100000
			then break
		end
	end
	output = torch.Tensor(output)
	
	fullset = {
		size = 100000,
		data=torch.Tensor(input),
		label=output
		}	
    return fullset
end


fullset=create_data("test.txt")

validationset = {
    size = 20000,
    data = fullset.data[{{80001,100000}}]:double(),
    label = fullset.label[{{80001,100000}}]
}

model1 = torch.load('model.net')
eval1 = function(dataset)
	local table={}
    local total = 0
    for i = 1,dataset.size do
        local output = model1:forward(dataset.data[i])
	
        local target= dataset.label[i]
	output[1]=output[1]*target[1]
	table[i]=output[1]
        local deviation = math.abs(output[1]-target[1])
        local percent = deviation/target[1]
        total = total + percent
      --local _, index = torch.max(output, 1) -- max index
      --local digit = index[1] % 10
      --if digit == dataset.label[i] then count = count + 1 end
   end
gnuplot.plot({torch.cat(dataset.label,torch.Tensor(table),2),'.'},{torch.cat(dataset.label,dataset.label,2),'.'})
   return total / dataset.size
end
print(100-eval1(validationset))

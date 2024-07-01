require 'torch'
require 'nn'
require 'optim'
require 'math'
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
--print(fullset)
trainset = {
    size = 80000,
    data = fullset.data[{{1,80000}}]:double(),
    label = fullset.label[{{1,80000}}]
}

--print(trainset.data)
validationset = {
    size = 20000,
    data = fullset.data[{{80001,100000}}]:double(),
    label = fullset.label[{{80001,100000}}]
}
--print(validationset.label)
--print(validationset)
model = nn.Sequential()
model:add(nn.Reshape(3))
model:add(nn.Linear(3,30))
model:add(nn.Tanh())
model:add(nn.Linear(30,30))
model:add(nn.Tanh())
model:add(nn.Linear(30, 1))
model:add(nn.Tanh())

criterion = nn.MSECriterion()
sgd_params = {
   learningRate = 0.001,
   learningRateDecay = 1e-4,
   weightDecay = 1e-7,
   momentum = 1e-3
}
x, dl_dx = model:getParameters()

step = function(batch_size)
    local current_loss = 0
    local count = 0
    local shuffle = torch.randperm(trainset.size)
    batch_size = batch_size or 200
    
    for t = 1,trainset.size,batch_size do
        -- setup inputs and targets for this mini-batch
        local size = math.min(t + batch_size - 1, trainset.size) - t
        local inputs = torch.Tensor(size,3)
		--print(inputs)
        local targets = torch.Tensor(size)
--		print(targets)
        for i = 1,size do
            local input = trainset.data[shuffle[i+t]]
            local target = trainset.label[shuffle[i+t]]
            
            inputs[i] = input
            targets[i] = target
        end
        --targets:add(1)
	--print(inputs)
        
        local feval = function(x_new)
            -- reset data
            if x ~= x_new then x:copy(x_new) end
            dl_dx:zero()

            -- perform mini-batch gradient descent
            local loss = criterion:forward(model:forward(inputs), targets)
            model:backward(inputs, criterion:backward(model.output, targets))

            return loss, dl_dx
        end
        
        _, fs = optim.sgd(feval, x, sgd_params)
        -- fs is a table containing value of the loss function
        -- (just 1 value for the SGD optimization)
        count = count + 1
        current_loss = current_loss + fs[1]
    end

    -- normalize loss
    return current_loss / count
end

eval = function(dataset, batch_size)
    local count = 0
    batch_size = batch_size or 200
    
   for i = 1,dataset.size,batch_size do
        local size = math.min(i + batch_size - 1, dataset.size) - i
        local inputs = dataset.data[{{i,i+size-1}}]
	       --print(inputs)
        local targets = dataset.label[{{i,i+size-1}}]
		--print(targets)
        local outputs = model:forward(inputs)
	for j=1,size do
		local a=targets[j]-outputs[j]
		--print(a)
		if a[1]<=0.2 then
		count=count+1
		end
	end	
	--print(targets)        
	--print(outputs)
    end

    return count/dataset.size
end
max_iters =30
do
    local last_accuracy = 0
    local decreasing = 0
    local threshold = 1 -- how many deacreasing epochs we allow
    for i = 1,max_iters do
        local loss = step()
        print(string.format('Epoch: %d Current loss: %4f', i, loss))
        local accuracy = eval(validationset)
        
        if accuracy < last_accuracy then
            if decreasing > threshold then break end
            decreasing = decreasing + 1
        else
            decreasing = 0
        end
        last_accuracy = accuracy
    end
end

paths = require 'paths'
filename = paths.concat(paths.cwd(), 'model.net')
torch.save(filename, model)	

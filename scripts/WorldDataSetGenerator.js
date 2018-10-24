var fs = require('fs');
const execFile  = require('child_process').execFileSync;
var countries = JSON.parse(fs.readFileSync(process.argv[2]));
var region_size = process.argv[3];
var width = process.argv[4];
var depth = process.argv[5];


//Generate datasets per country
for(country in countries){
	const dsg = execFile('node', 
			['scripts/DataSetProducer.js',
			countries[country],				//region to generate dataset
			region_size,						//size of region
			width,								//width
			depth],			 				 	//depth
		 	{stdio:[0,1,2]}					//redirect output from parent to child process
	);
}


// ##################################################################
// This script finds most popular videos for a region.
// Then applies BFS traversal for each most popular video, and constructs
// a graph with all the necessary related videos.

// Region, region size, width and depth are provided via the args list.
// ##################################################################

var region = process.argv[2];
var region_size = process.argv[3];
var width = process.argv[4];
var depth = process.argv[5];

var authoriz = require('./authorization.js');
var fs = require('fs');
var {google} = require('googleapis');
const execFile  = require('child_process').execFileSync;

// First of all we calculate the most popular videos per region(provided by the command line).
const fmp = execFile('node', 
			['scripts/findMostPopularVideos.js',
			region,							//region to calculate most popular videos from
			region_size],	 				//size of mp videos
		 	{stdio:[0,1,2]}					//redirect output from parent to child process
);

var date = String(new Date());
date = date.split(" ");
date = date[2] + "_" + date[1] + "_" + date[3];

// We collect all the most popular videos per region to pass them as seeds to the BFS traversal script.
var regional_dataset = [];
var mp = JSON.parse(
	fs.readFileSync("Results\\Most_Popular_Videos\\" + region_size + "_mostPopular_" + region + "_(" + date + ").json")
);

for(id in mp){
	const bfs = execFile('node', 
			['scripts/BFStraversal.js',
			String(mp[id]),					//content id to apply bfs on
			width,							//size of related videos per id
			depth],	 						//depth of bfs algorithm
		 	{stdio:[0,1,2]}					//redirect output from parent to child process
	);
	// We collect all the datasets per id(most popular videos) and place them in the regional dataset list
	id_based_dataset = JSON.parse(
				fs.readFileSync("Results/DataSets_perID/seedID_" + String(mp[id]) + "_width_" + width + "_depth_" + depth + "_(" + date + ").json")
	);
	for(id in id_based_dataset){
		regional_dataset.push(id_based_dataset[id]);
	}
}

// The final graph will be an array consisted of two lists. The first list will be the most popular videos over which the 
// BFS traversal will be applied. The second list will be the result of bfs traversal over the most popular videos.
var final_graph = {};
final_graph["most_popular"] = mp;
final_graph["edge_list"] = regional_dataset;

// We write the regional dataset list to a file.
fs.writeFile("Results\\DataSets_perRegion\\Region_" + region + "_regionSize_" + region_size + "_width_" + width + "_depth_" + depth + "_(" + date + ").json", JSON.stringify(final_graph, null, "\t"), function (err) {
            if (err) {
                return console.log(err);
            }
});
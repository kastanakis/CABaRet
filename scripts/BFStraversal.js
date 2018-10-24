// ##################################################################
// This script applies BFS traversal algorithm over a single video ID.
// The result is a graph with depth d and width w that contains related videos from
// every depth. Every new depth contains the related videos of the 
// previous depth results.

// Workflow: 
// Width, depth and seed ID are provided via the args list.
// The script requests w videos related to the seed ID and adds them to the graph
// in the order they are returned from the YouTube API.For each video w returned,
// the w' related videos of that are calculated, so on so forth, until depth d is reached,
// and the graph is filled with all necessary related videos.

// ##################################################################


var content_id = process.argv[2];
var width = process.argv[3];
var depth = process.argv[4];

var authoriz = require('./authorization.js');
var fs = require('fs');
var {google} = require('googleapis');
const execFile  = require('child_process').execFileSync;

var date = String(new Date());
date = date.split(" ");
date = date[2] + "_" + date[1] + "_" + date[3];

// We create a global list to hold all related videos in the order they are returned
// from the API. This list will contain all video IDs from all depths, provided via
// a seed ID.
var graph_of_ID = [];
// We also need a list that will be the seed ID list for the related videos function.
// This list will contain all contents to be traversed, thus their related lists will be
// calculated. This list gets replaced before every new depth iteration with the related of the current
// depth keys. The replacement occurs by collecting the values of each ID that the findRelatedVideos 
// function has been applied upon. This values are saved under a specific directory and are easily acce-
// sible. The files are saved containing the name of the id that they have applied FRV on.
var Contents_to_be_traversed = []; 
// We also need a temporary list for the proper filling of Contents_to_be_traversed.
// This list collects every ID used in the findRelatedVideos function, in order for the file indexing
// to be easier.
var previous_depth_temp_list = [];
// This counter holds the current depth.
var depth_counter = 0;
// At first we populate the traversing list with the seed ID.
Contents_to_be_traversed.push(content_id);

while(depth_counter < depth){
	// Clear the indexing list. We collect each id that the FRV function is applied on at this depth.
	// Thats why we clear it at the beginning of each iteration.
	previous_depth_temp_list = [];
	// In the end of the following loop, there will be files created under a specific directory(Results/Related_to_ID_Videos/)
	// containing related videos.
	for( videoID in Contents_to_be_traversed){
		content_id = Contents_to_be_traversed[videoID];
		// We execute the FRV function over each key from the Contents to be traversed list.
		if(!fs.existsSync("Results/Related_to_ID_Videos/" + width + "_Related_to_" + content_id + "_(" + date + ").json")){
			const bfs = execFile('node', 
				['scripts/findRelatedVideos.js',
				content_id,						//content id to calculate related videos on
				width],	 						//size of related videos
			 	{stdio:[0,1,2]}					//redirect output from parent to child process
			);
		}
		// Also keep every id to help us do the file indexing and populate the Contents to be traversed list easier.
		previous_depth_temp_list.push(content_id);
	}
	
	// Before we proceed to the new depth level we have to populate the Contents_to_be_traversed_list with the values of the related
	// videos calculated in the previous FRV function.
	Contents_to_be_traversed = [];
	for(temp_content in previous_depth_temp_list){
		//get the values of the current keys been examined, and set them as keys for the next depth level
		set_keys_for_next_depth_level = JSON.parse(
			fs.readFileSync("Results/Related_to_ID_Videos/" + width + "_Related_to_" + previous_depth_temp_list[temp_content] + "_(" + date + ").json")
		);
		//create a key value object to push it to the final graph that will be our Dataset
		var obj = {};
		//set as key the id used as seed to the findRelated script and as value the list of related(extracted from the file)
		obj[previous_depth_temp_list[temp_content]] = set_keys_for_next_depth_level;

		//prepare the contents_to_be_traversed list for the next depth level
		for(key in set_keys_for_next_depth_level){
			Contents_to_be_traversed.push(set_keys_for_next_depth_level[key]);
		}

		// We collect the related videos in a graph in the order they are returned from the API.
		graph_of_ID.push(obj);
	}
	// Move to the next depth level
	depth_counter++;
}	


// We write the BFS traversing result to a file.
fs.writeFile("Results\\DataSets_perID\\seedID_" + process.argv[2] + "_width_" + width + "_depth_" + depth + "_(" + date + ").json", JSON.stringify(graph_of_ID, null, "\t"), function (err) {
            if (err) {
                return console.log(err);
            }
});						

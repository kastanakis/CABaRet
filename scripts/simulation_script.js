// ##################################################################
// This script returns a bitmap of cache hits or misses based on
// a series of specific amount of simulation runs.
// For sim_runs iterations we calculate the recommendation list(based on
// BFStraversal) of a given seed id.
// We choose a random video as the new seed id for the new iteration(randomness is based on the
// possibility mode provided). 
// If the chosen video is a cache hit we score an 1 on a bitmap size of sim_runs.
// If the chosen video is a cache miss we score an 0 on a bitmap size of sim_runs.

// ##################################################################

var fs = require('fs');
const execFile  = require('child_process').execFileSync;

var seed_id = process.argv[2];
var width = process.argv[3];
var depth = process.argv[4];
var recommendation_size = process.argv[5];
var cached_contents = JSON.parse(fs.readFileSync(process.argv[6]));
var sim_runs = process.argv[7];
var selection_mode_possibilities = process.argv[8];

var date = String(new Date());
date = date.split(" ");
date = date[2] + "_" + date[1] + "_" + date[3];

// Calculate possibilities for each related in final list based on the selection probability
function weights_array(recommendation_size, selection_mode_possibilities){
	var sigma = 0;
	var sigma_pow2 = 0;
	var beta = 0;
	var beta_pow2 = 0;

	var weights = [];
	var weights_pow2 = [];
	var regular = [];

	for(var i = 1; i <= recommendation_size; i++){
		sigma += 1/i;
		wtf = Math.pow(i,2);
		sigma_pow2 += (1/wtf);
	}
	beta = 1/sigma;
	beta_pow2 = 1/sigma_pow2;
	for(var j = 1; j <= recommendation_size; j++){
		weights.push(beta/j);
		wtf = Math.pow(j,2);
		weights_pow2.push(beta_pow2/wtf);
		regular.push(1/recommendation_size);
	}
	if(selection_mode_possibilities == "zipf1") return weights;
	else if(selection_mode_possibilities == "zipf2") return weights_pow2;
	else return regular;
}

// Calculate the graph of a given id with specific width and depth
function bfs_traversal(seed_id, width, depth){
	var dataset = [];
	// Calculate the BFS graph from the seed ID
	const bfs = execFile('node', 
				['scripts/BFStraversal.js',
				String(seed_id),				//content id to apply bfs on
				width,							//size of related videos per id
				depth],	 						//depth of bfs algorithm
			 	{stdio:[0,1,2]}					//redirect output from parent to child process
	);

	// We collect the dataset per id (from the execution of BFStraversal)
	var id_based_dataset = JSON.parse(
		fs.readFileSync("Results/DataSets_perID/seedID_" + String(seed_id) + "_width_" + width + "_depth_" + depth + "_(" + date + ").json")
	);

	// We do the necessary unlisting and unboxing of the json object, so that 
	// the total graph will be handled as a single list.
	// We parse the id_based_dataset and collect only the values of the json object.
	for(id in id_based_dataset){
		var temp = Object.values(id_based_dataset[id])[0];
		for(idd in temp){
			// We trim of the possible undefined values from the unlisting.
			if(typeof(temp[idd]) === 'string'){
				dataset.push(temp[idd]);
			}
		}
	}
	// Return a list with the related videos graph
	return dataset;
}

Array.prototype.diff = function(a) {
    return this.filter(function(i) {return a.indexOf(i) < 0;});
};

// Calculate the recommendation list of a given id based on the cached contents
function recommendation_list(recommendation_size, cached_contents, dataset, seed_id){
	// Spot cached content from the previous graph
	var cache_hits = [];
	for(cached in cached_contents){
		// From the list(bfs graph) provided by the bfs traversal function, we extract
		// the ids that are existent in the cached contents list, to the cache_hits list.
	    if (dataset.includes(cached_contents[cached])){
	        cache_hits.push(cached_contents[cached]);
	    }
	}
	// Remove duplicates from cache_hits list
	var cache_hits_set = new Set(cache_hits);
	// Remove initial content(seed id) from cache hits set. We dont want to recommend
	// the currently watching video again to the user.
	// The cache_hits list will be a part of the recommendation list.
	if(cache_hits_set.has(seed_id)) cache_hits_set.delete(seed_id);
	// console.log("Cache hits");
	// console.log(cache_hits_set);

	// Fill the rest recommendation list with legacy youtube related videos
	var remaining_contents_number = recommendation_size - cache_hits_set.size;
	// If the recommendation list populated by cache_hits_list has filled, return 
	// only the requested size.
	if(remaining_contents_number <= 0) {
		return Array.from(cache_hits_set).slice(0, recommendation_size);
	}// if not filled, populate the rest of the list with legacy youtube ids recommended for the initial content
	else {
		// Fill the rest recommendation set with legacy youtube videos
		legacy_ids = JSON.parse(
			fs.readFileSync("Results/Related_to_ID_Videos/" + width + "_Related_to_" + seed_id + "_(" + date + ").json")
		);

		// Return an array with first occurences to be the cache hits set and then collect
		// the difference between legacy set and cache set(legacy set minus the intersection with cache set)
		// and append(concat) it to the cache set. This array must be of size = recommendation_size
		return Array.from(cache_hits_set).concat(legacy_ids.diff(Array.from(cache_hits_set))).slice(0, recommendation_size);
	}

}

function choose_random_video_from_related(related, weights){
	var freq_array = [];
	//construct freq array depending on weights and related set
	for(weight in weights){
		for(ii = 0; ii < Math.round(weights[weight] * 100); ii++){
			freq_array.push(related[weight]);
		}
	}
	//pick random element from freq_array
	var to_return = freq_array[Math.floor(Math.random()*freq_array.length)];
	return to_return;	
}

// Main Function
// We create the possibilities array
// For sim_runs iterations we calculate the recommendation list of the current seed id(via calling the recommendation_list func
// over the bfs_traversal graph provided).
// We then apply the necessary possibility over each recommended id, to simulate the end user click over a random video id from
// the recommendation list.
// The chosen video will be the seed id for the new iteration. 
// If the chosen video is a cache hit we score an 1 on a bitmap size of sim_runs.
// We return a struct with every recommendation list at each step plus the random selection of each step plus the total bitmap.
function main(recommendation_size, selection_mode_possibilities, sim_runs, cached_contents, seed_id, width, depth){
	var possibilities_for_selection = weights_array(recommendation_size, selection_mode_possibilities);
	var cache_hits_bitmap = [];
	var return_object = {};
	for(i = 0; i < sim_runs; i++){
		// We apply the possibilities array over the recommendation list
		// We pick the next seed_id based on the possibilities applied on the recommendation list.
		recommendationL = recommendation_list(
				recommendation_size, 
				cached_contents, 
				bfs_traversal(
					seed_id, 
					width, 
					depth), 
				seed_id
		);
		// console.log("Recommendation list");
		// console.log(recommendationL);
		return_object["RecommendationList_step" + String(i+1)] = recommendationL;
		seed_id = choose_random_video_from_related(
			recommendationL, 
			possibilities_for_selection
		);
		// console.log("Random selection");
		// console.log(seed_id);
		return_object["RandomSelection_step" + String(i+1)] = seed_id;
		if(cached_contents.includes(seed_id)){
			cache_hits_bitmap.push(1);
		} else {
			cache_hits_bitmap.push(0);
		}
		// console.log("Cache hits bitmap");
		// console.log(cache_hits_bitmap);
	}
	return_object["CacheHitsBitmap"] = cache_hits_bitmap;
	console.log(return_object);
	return return_object;
}


var Bitmap = main(recommendation_size, selection_mode_possibilities, sim_runs, cached_contents, seed_id, width, depth);
fs.writeFile("Results\\Cache_Bitmaps\\SeedID_" + seed_id + "_width_" + width + "_depth_" + depth + "_mode_" + selection_mode_possibilities +"_recommendationSize_" + recommendation_size + "_(" + date + ").json", JSON.stringify(Bitmap, null, "\t"), function (err) {
            if (err) {
                return console.log(err);
            }
});

// ##################################################################
// This script returns bitmaps of cache hits or misses based on
// a series of specific amount of simulation runs over a region.
// Cache hit is scored as an 1, while cache miss is scored as a 0 in
// a bitmap.
// This procedure takes place on a list of most popular videos from
// a region. This list of most popular is considered to be the cache 
// list as well.
// Results are written under the Results/Cache_Bitmaps directory.

// ##################################################################

var fs = require('fs');
const execFile  = require('child_process').execFileSync;

var region = process.argv[2];
var region_size = process.argv[3];
var width = process.argv[4];
var depth = process.argv[5];
var sim_runs = process.argv[6];
var mode = process.argv[7];


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
cache_url = "Results/Most_Popular_Videos/" + region_size +"_mostPopular_" + region + "_(" + date + ").json"

var most_pplr = JSON.parse(fs.readFileSync(cache_url));
console.log(region_size + " Most Popular Videos in " + region);
console.log(most_pplr);

// For each most popular id execute a series of simulation script runs.
// In each simulation the recommendation list of a given seed id is 
// calculated, and based on an array of possibilities the next seed id
// is picked from this recommendation list.
// This procedure is called for every most popular video in the most_pplr list.
// The result for each bitmap is an array of 1 and 0, 1 indicating a cache hit
// and 0 indicating a cache miss.
for(mp in most_pplr){
	const sim_script = execFile('node', 
				['scripts/simulation_script.js',
				most_pplr[mp],					//seed id 
				width,							//width
				depth,							//depth
				region_size,					//size of recommendations(same as region size since region's mp = cache)
				cache_url,						//cache contents
				sim_runs,						//sim runs
				mode],	 						//randomness mode
			 	{stdio:[0,1,2]}					//redirect output from parent to child process
	);
}

// The results are writen under the Results/Cache_Bitmaps directory.

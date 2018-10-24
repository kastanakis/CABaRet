// ##################################################################
// This script finds most popular videos of a given region and writes
// them to a file.

// Region and region size are provided via the args list.
// ##################################################################

// DEFINE THE REGION TO EXTRACT MOST POPULAR VIDEOS FROM
var region = String(process.argv[2]);
// DEFINE THE NUMBER OF MOST POPULAR VIDEOS TO BE EXRACTED
var results_slice = String(process.argv[3]);
// Check if args of user are in bounds
if((isNaN(parseInt(results_slice)) || parseInt(results_slice) > 50 || parseInt(results_slice) < 1)){
    results_slice = "50"; 
} 
var authoriz = require('./authorization.js');
var fs = require('fs');
var {google} = require('googleapis');
var obj = {};


// Load client secrets from a local file.
// Authorizes the function "videosListMostPopular" granting a client secret key,
// so we can request the top most popular videos for a specific region from the 
// YouTube API.
fs.readFile('client_secret.json', function processClientSecrets(err, content) {
    if (err) {
        console.log('Error loading client secret file: ' + err);
        return;
    }
    // Authorize a client with the loaded credentials, then call the YouTube API.
    authoriz.data.authorize(JSON.parse(content), {
        'params': {
            'chart': 'mostPopular',
            'regionCode': region,
            'maxResults': results_slice,
            'part': 'snippet',
            'fields' : 'items(id)'
        }
    }, videosListMostPopular);

});

// Requests the top most popular videos for a specific region.
// Saves the output list under the Results/Most_Popular_Videos directory as a json object.
function videosListMostPopular(auth, requestData) {
    var service = google.youtube('v3');
    var parameters = authoriz.data.removeEmptyParameters(requestData['params']);
    parameters['auth'] = auth;
    service.videos.list(parameters, function (err, response) {
        if (err) {
            console.log('The API returned an error: ' + err);
            return;
        }
        var relatedList = [];
        for (mostPopular in response.data.items) {
            relatedList.push(response.data.items[mostPopular].id);
        }
        var date = String(new Date());
        date = date.split(" ");
        date = date[2] + "_" + date[1] + "_" + date[3];
        fs.writeFile("Results\\Most_Popular_Videos\\" + results_slice + "_mostPopular_" + region + "_(" + date + ").json", JSON.stringify(relatedList, null, "\t"), function (err) {
            if (err) {
                return console.log(err);
            }
        });
    });
}
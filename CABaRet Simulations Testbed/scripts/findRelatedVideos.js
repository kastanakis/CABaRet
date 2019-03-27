// ##################################################################
// This script finds related videos of a given ID and writes
// them to a file.

// Video id and recommendation list size are provided via the args list.
// ##################################################################

// DEFINE THE VIDEO ID TO EXTRACT THE RELATED LIST FROM
var input_data = process.argv[2];
// DEFINE THE SIZE OF RELATED VIDEOS TO BE EXTRACTED
var output_size = process.argv[3];
// Check if args of user are in bounds
if((isNaN(output_size) || output_size > 50 || output_size < 1)){
    output_size = 50; 
} 
var authoriz = require('./authorization.js');
var fs = require('fs');
var {google} = require('googleapis');
var videoId = String(input_data);

// Load client secrets from a local file.
// Authorizes the function "searchListRelatedVideos" granting a client secret key,
// so we can request the related videos for a specific video id from the 
// YouTube API.
fs.readFile('client_secret.json', function processClientSecrets(err, content) {
    if (err) {
        console.log('Error loading client secret file: ' + err);
        return;
    }

    //Request related videos for video provided from the args
    authoriz.data.authorize(JSON.parse(content), videoId, searchListRelatedVideos);
    
});

// Requests related videos of a specific video ID.
// Saves the output list under the Results/Related_to_ID_Videos directory as a json object.
function searchListRelatedVideos(auth, videoId) {
    var service = google.youtube('v3');
    var requestData = {
        'params': {
            'part': 'snippet',
            'fields' : 'items(id)',
            'relatedToVideoId': videoId,
            'type': 'video',
            'maxResults': output_size
        },
    };
    var parameters = authoriz.data.removeEmptyParameters(requestData['params']);
    parameters['auth'] = auth;
    service.search.list(parameters, function (err, response) {
        if (err) {
            console.log('The API returned an error: ' + err);
            return;
        }
        //populate the related list
        var relatedList = [];
        for (item in response.data.items) {
            relatedList.push(response.data.items[item].id.videoId);
        }
        var date = String(new Date());
        date = date.split(" ");
        date = date[2] + "_" + date[1] + "_" + date[3];
        fs.writeFile("Results\\Related_to_ID_Videos\\" + output_size + "_Related_to_" + input_data + "_(" + date + ").json", JSON.stringify(relatedList, null, "\t"), function (err) {
            if (err) {
                return console.log(err);
            }
        });
    });
}
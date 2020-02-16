/* Each experiment is marked by a unique sessionID */
var sessionID = '';

function logEverything(legacyYouTubeListGlobal, watchingID){
    var toReturn = {
        'VideoID' : watchingID,
        'YoutubeLegacyRecommendationList' : legacyYouTubeListGlobal,
        'CachedContents' : CACHE,
        'OfferedRecommendationList' : collectRecList(),
        'Ratings' : {
            'VideoQuality' : $('input[name=qual]:checked').val(),
            'RecommendationQuality' : $('input[name=rec]:checked').val(),
            'VideoInterest' : $('input[name=interest]:checked').val(),
            'OverallEnjoyment' : $('input[name=general]:checked').val(),
            'OptionalComments' : document.getElementById("optional").value
        },
        'VideoTimestamp' : String(new Date()),
        'ExperimentTimestamp' : experiment_timestamp,
        'CacheHit' : CACHE.includes(watchingID),
        'RecommendationAlgorithm' : 'CABaRet',
        'Region' : window.location.href.split("region=")[1].split("#")[0],
        'ExperimentNum' : window.location.href.split("exp")[1].split("/")[0],
        'Interruptions' : Interruptions
    };
    
    /* Log per experiment */
    $.ajax({ 
		type: "POST",
		url: '../logger.php' , 
		data: {'videoSession': toReturn, 'sessionID' : sessionID} ,
        datatype: "json",
        success: function(data){
            /* need to keep track of the pseudo random logfile name */
            sessionID = data;
        }});
    
}

function collectRecList(){
    var reclist = [];
    $.each($('#results')[0].childNodes, function(i, item){
        reclist.push(item.childNodes[1].src.split('vi/')[1].split('/')[0]);
    })
    return reclist;
}


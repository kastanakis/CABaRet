/* That's the recommendation module we provide on CABaRet.
Feel free to adjust or remove this implementation and add YOUR OWN,
to test how it performs within this platform. You
can find more information about YouTube API in their documentation:
https://developers.google.com/youtube/v3/  */
function cabaret_implementation(id) {
    /* We calculate the rec list */
    $.ajax({ 
		type: "GET",
		url: '../relatedVideos.php' , 
		data: {"id": id, "maxResults": 50} ,
		datatype: "json",
		success: function (data) {
            /* Holds the video ids so that we wont insert a duplicate video in the reclist */
            var cacheHits = [];
            var counter = 0;
            legacyYouTubeListGlobal = [];
            data = JSON.parse(data);
            /* We clear the recommendation list */
            $('#results').empty();

            /* We hold the appropriate videos and append them to the recommendation list (depth1)*/
            $.each(data.items, function (i, item) {
                legacyYouTubeListGlobal.push(item.id.videoId);
                //Collect the contents that are:
                if (item.id.videoId != id && /*not currently watched by the user (we dont recommend the current streaming video)*/
                    CACHE.includes(item.id.videoId) && /*existent in our cache list (we collect only possible cache hits)*/
                    !cacheHits.includes(item.id.videoId) /*not a duplicate (we filter out ids that are existent already in the rec list)*/
                    /*&&CACHE_frequency[item.id.videoId] == 0*/) { /*there must be fairness between cached contents. serve a video only with occurence 0*/
                    cacheHits.push(item.id.videoId);
                    // CACHE_frequency[item.id.videoId] = 1;
                    counter++;
                    /* Create a thumbnail for each of those recommendations */
                    reclist = '<li class=\"list-group-item\" onclick=\"atLaunch(\'' +
                        item.id.videoId + '\')\"><h4>' + item.snippet.title + '</h4>' +
                        '<img src="' +
                        item.snippet.thumbnails.medium.url + '"></a>' + '</li>';
                        /* Fill only the necessary amount of recommendations in the list */
                    if (counter <= 5) $('#results').append(reclist);
                }
            })

            /* We hold all return values of ajax requests to use as promises later on */
            var gets = [];
            /* We calculate the depth only of the 10 first related videos, not all 50 */
            data.items = data.items.slice(0,10);

            /* We calculate depth 2 */
            $.each(data.items, function (i, item) {
                gets.push($.ajax({ 
                    type: "GET",
                    url: '../relatedVideos.php' , 
                    data: {"id": id, "maxResults": 50} ,
                    datatype: "json",
                    success: function (data) {
                        data = JSON.parse(data);
                        /* Hold the appropriate videos and append them to the recommendation list*/
                        $.each(data.items, function (i, item) {
                            //Collect the cache hits that are:
                            if (item.id.videoId != id && /*not currently watched by the user (we dont recommend the current streaming video)*/
                                CACHE.includes(item.id.videoId) && /*existent in our cache list (we collect only possible cache hits)*/
                                !cacheHits.includes(item.id.videoId) /*not a duplicate (we filter out ids that are existent already in the rec list)*/
                                /*&&CACHE_frequency[item.id.videoId] == 0*/) { /*there must be fairness between cached contents. serve a video only with occurence 0*/
                                cacheHits.push(item.id.videoId);
                                // CACHE_frequency[item.id.videoId] = 1;
                                counter++;
                                /* Create a thumbnail for each of those recommendations */
                                reclist = '<li class=\"list-group-item\" onclick=\"atLaunch(\'' +
                                    item.id.videoId + '\')\"><h4>' + item.snippet.title + '</h4>' +
                                    '<img src="' +
                                    item.snippet.thumbnails.medium.url + '"></a>' + '</li>';
                                    /* Fill only the necessary amount of recommendations in the list */
                                if (counter <= 5) $('#results').append(reclist);
                            }
                        })
                    }}))
            })

            /* Now that cached content is prioritized in the top of the reclist we can append the legacy youtube list in the bottom*/
            /* "when" keyword is used for promises in JS */
            $.when.apply($, gets).then(function () {
                /* When we have finished with the collection of cached contents on top of the reclist
                we need to fill any gaps left in the recommendation list due to not sufficient number of contents */
                /* For each data collected in the first depth calculation (equals to legacy youtube videos normally suggested by YouTube service)*/
                $.each(data.items, function (i, item) {
                    /* We append youtube videos until reclist has size 5*/
                    /* We dont use videos that are duplicates nor videos equal with current watching video*/
                    /* We create a thumbnail for those ids */
                    if (counter < 5 && item.id.videoId != id && !cacheHits.includes(item.id.videoId)) {
                        reclist = '<li class=\"list-group-item\" onclick=\"atLaunch(\'' +
                            item.id.videoId + '\')\"><h4>' + item.snippet.title + '</h4>' +
                            '<img src="' +
                            item.snippet.thumbnails.medium.url + '"></a>' + '</li>';
                        $('#results').append(reclist);
                        counter++;
                    }
                })
                /* Change what back button does in Browsers */
	            history.pushState({}, '', '');

            })
        }})
        /* Returns the update CACHE_frequency mapping */
        // return CACHE_frequency;
}
/* Include a .js file in another .js file with jquery */
$.getScript("../JavaScript/player.js", function () {});
$.getScript("../JavaScript/logger.js", function () {});
$.getScript("../JavaScript/CABaRet.js", function () {});

/* host server name */
// var hostname = 'http://localhost:8080/';
var hostname = 'http://cabaret.ddns.net/';

/* CACHE contains all video ids that represent the cached contents in our experiment */
var CACHE = [];
/* We collect the region provided (by users) in the first stages of the experiment */
var region = window.location.search.substring(1).split("=")[1];
/* We need this for the click and load functions to be global, used as a temporary var */
var legacyYouTubeListGlobal = [];
/* player is an object of a singleton class, thus one of a kind in our program, hence needs
to be global */
var player;
/* number of video sessions before giving ticket */
var simulation_runs = 5;
/* need to hold the timestamp of each experiment in logs */
var experiment_timestamp = '';

/* Change what back button does in browsers */
window.onpopstate = function (event) {
	window.location.href = hostname;
};

/* When page loads...*/
$(document).ready(function (event) {
	/* We collect the Cached contents from the file system */
	$.ajax({
		url: hostname + 'Resources/Caches/500_mostPopular_' + region + '.json',
		dataType: 'json',
		success: function (response) {
			CACHE = response;
			/* We offer the Most Popular videos atm in the recommendations list of 
			the experiment. Those video ids correspond to the most popular videos from the
			selected region by the user. */
			mostpopularids = CACHE.slice(0, 50);
			/* We provide different list of most popular videos on each experiment */
			ditems = shuffle(mostpopularids).slice(0, 20);
			/* Do something with the content you received from the YouTube API */
			$.each(ditems, function (i, id) {
				var title = '';
				$.ajax({
					/* Extract info for video from id */
					url: 'https://noembed.com/embed?url=https://www.youtube.com/watch?v=' + id,
					success: function (data) {
						title = JSON.parse(data).title;
						if (title != undefined) {
							mostPop = '<u><li class=\"list-group-item\" onclick=\"instantiatePlayer(\'' +
								id + '\');atLaunch(\'' +
								id + '\')\"><h4>' + title + '</h4>' +
								'<img src="' +
								'http://img.youtube.com/vi/' + id + '/mqdefault.jpg' + '"></a>' + '</li></u>';
							/* We append those videos (response) in the recommendation list */
							$('#results').append(mostPop);
						}
					}
				});
			})
		}
	});
	/* Show guidelines to the user */
	confirm(`Dear user,\n\nThis experiment is part of a research effort by researchers at the Institute of Compute Science, Foundation of Research and Technology Hellas (ICS-FORTH), Greece.\n\nThe experiment uses a platform based on the YouTube online video service and API. Its goal is to study the interplay between video QoS, user QoE, and quality of recommendations, and understand and evaluate the user QoE for cache-aware recommendation systems.\nYou are requested to enter the platform, select and watch a series of five (5) YouTube videos, and rate them and their lists of related videos. You may experience interruptions while watching some videos; some of them may be artificially introduced for the purposes of this research study.\nEach experiment may take approximately 5 minutes.\nNo personal information is requested in the experiment. All data will be anonymous, and collected and published in a manner that would not allow identification of your personal identity.\n\nFor further information, questions, etc., feel free to contact us at:\nPavlos Sermpezis, sermpezis@ics.forth.gr, Post-doctoral Researcher, FORTH, Greece\nSavvas Kastanakis, kastan@csd.uoc.gr, Master Student, FORTH and University of Crete, Greece`);
})

/* Fetch the video to the player and calculate the rec list */
function atLaunch(id) {
	/* Clear previous intervals (lags caused in non cached contents) */
	clearInterval(intervalID);

	if ($('#player').attr('src') != undefined) {
		/* We collect the video id of the current video session to log its relevant information */
		var previousWatchingID = $('#player').attr('src').split('https://www.youtube.com/embed/')[1].split('?')[0];
		/* When a new video is requested we need to log all previous information about the ratings and relevant information */
		logEverything(legacyYouTubeListGlobal, previousWatchingID);
	} else {
		/* Timestamp of experiment launch */
		experiment_timestamp = new Date();
	}

	/* Fetch the video to the player */
	var url = "https://www.youtube.com/embed/" + id + "?autoplay=1&origin=" + hostname + "&enablejsapi=1&controls=1";
	$('#player').attr('src', url);

	/* We update/refresh the html */
	reset_and_enable_ratings();

	/* When simulation_runs == 0 we end the experiment */
	if (!simulation_runs--) {
		/* Change what back button does in Browsers */
		history.pushState({}, '', '');
		offerTicket(sessionID);
	} else {
		/* Corrupt the quality of non cached videos */
		if (CACHE.includes(id) == false) {
			corruptQualityOfExperience(id);
		}
		/* We offer our Cache Aware BFS Recommendation module */
		cabaret_implementation(id);
	}
}

/* We reset our ratings interface so we can collect fresh values in the new sessions */
function reset_and_enable_ratings() {
	/* Reset the ratings from the previous session */
	document.getElementById("disRating").reset();
	document.getElementById("recRating").reset();
	document.getElementById("qualRating").reset();
	document.getElementById("generalRating").reset();
	document.getElementById("optional").value = '';

	/* Enable the ratings only the first time, since in the beginning of the experiment,
	they are ALL set to disabled, thus we check iff one is set to disabled and we enable all if true */
	if (document.getElementById("star-1").disabled == true) {
		document.getElementById("star-1").disabled = false;
		document.getElementById("star-11").disabled = false;
		document.getElementById("star-2").disabled = false;
		document.getElementById("star-21").disabled = false;
		document.getElementById("star-3").disabled = false;
		document.getElementById("star-31").disabled = false;
		document.getElementById("star-4").disabled = false;
		document.getElementById("star-41").disabled = false;
		document.getElementById("star-5").disabled = false;
		document.getElementById("star-51").disabled = false;
		document.getElementById("star-151").disabled = false;
		document.getElementById("star-141").disabled = false;
		document.getElementById("star-131").disabled = false;
		document.getElementById("star-121").disabled = false;
		document.getElementById("star-111").disabled = false;
		document.getElementById("star-515").disabled = false;
		document.getElementById("star-414").disabled = false;
		document.getElementById("star-313").disabled = false;
		document.getElementById("star-212").disabled = false;
		document.getElementById("star-101").disabled = false;
	}
}

function shuffle(arra1) {
	var ctr = arra1.length,
		temp, index;
	// While there are elements in the array
	while (ctr > 0) {
		// Pick a random index
		index = Math.floor(Math.random() * ctr);
		// Decrease ctr by 1
		ctr--;
		// And swap the last element with it
		temp = arra1[ctr];
		arra1[ctr] = arra1[index];
		arra1[index] = temp;
	}
	return arra1;
}

function offerTicket(sessionID) {
	$("body").children().replaceWith("<div style='margin-left: 30%;margin-top: 5%;padding: 10px;'><h1 style='color:red;'>Thank you for participating</h1><h5>For further information, questions, feedback, etc.,<br>feel free to contact us at:<li>Pavlos Sermpezis, sermpezis@ics.forth.gr, <br>Post-doctoral Researcher, FORTH, Greece</li><br><li>Savvas Kastanakis, kastan@csd.uoc.gr,</li> Master Student, University of Crete, Greece</h5></div>");
}
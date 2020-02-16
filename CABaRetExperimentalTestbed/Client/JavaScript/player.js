var public_id = '';
var intervalID = 0;
var startPoint = 0;
var delay = 10000;
var Interruptions = 0;
/**
 *
 *
 * @class Player
 * Returns a player object with play and pause functions, 
 * construction in a singleton patterb and event handling
 * of various states (ready, error, change ). 
 */
class Player {
	constructor(id) {
		// Player is a singleton class. There can only be one instance of that class at
		// each run. That's the reason why var player is global.
		if (typeof (id) != 'undefined') {
			player = new YT.Player('player', {
				videoId: String(id),
				playerVars: {
					'controls': 1,
					'autoplay': 1,
					'enablejsapi': 1,
					'origin': hostname
				},
				events: {
					'onReady': this.onPlayerReady,
					'onError': this.onPlayerError
				}
			});
		}
	}

	/* Called when player is ready to stream a video id */
	onPlayerReady(event) {
		// If you want autoplay in the first video occurence uncomment the below line
	}	

	/* Called when player crashes for some reason */
	onPlayerError(event) {
		console.log("ERROR IN PLAYER:" + event);
	}

	/* Called when player need to stop streaming. */
	playVideo() {
		// this.loadVideo();
	}
}

/* This function is called in the first loading of the experiment, where the player is instanstiated */
function instantiatePlayer(id) {
	if (typeof (YT) == 'undefined' || typeof (YT.Player) == 'undefined') {
		var tag = document.createElement('script');
		tag.src = "https://www.youtube.com/iframe_api";
		var firstScriptTag = document.getElementsByTagName('script')[0];
		firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
		window.onYouTubePlayerAPIReady = function () {
			public_id = id;
			new Player(id);
		};
	}
}

/* This function is called when we want to simulate a congested network */
function corruptQualityOfExperience(id) {
	clearInterval(intervalID);
	Interruptions = 0;
	startPoint = 0;
	intervalID = window.setInterval(function () {
		startPoint = startPoint + delay/1000 - 2;
		Interruptions++;
		var url = "https://www.youtube.com/embed/" + id + "?autoplay=1&origin=" + hostname + "&enablejsapi=1&controls=1&start=" + startPoint;
		$('#player').attr('src', url);	}
	, delay);
}
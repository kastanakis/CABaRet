<?php
    $API_URL = "https://www.googleapis.com/youtube/v3/search?";
    $parameters = [
        'part' => 'snippet',
        'type' => 'video',
        'maxResults' => $_GET['maxResults'],
        'relatedToVideoId' => $_GET['id'],
        'fields' => 'items(id, snippet(title, thumbnails(medium)))',
        'key' => 'YOU_MUST_HAVE_ONE'
    ];
    $response = file_get_contents($API_URL.http_build_query($parameters));
    echo $response;
?>
<?php
    $API_URL = "https://www.googleapis.com/youtube/v3/videos?";
    $parameters = [
        'part' => 'snippet',
        'regionCode' => $_GET['region'],
        'maxResults' => $_GET['maxResults'],
        'chart' => 'mostPopular',
        'fields' => 'items(id, snippet(title, thumbnails(medium), categoryId))',
        'key' => 'YOU_MUST_HAVE_ONE'
    ];
    $response = file_get_contents($API_URL.http_build_query($parameters));
    echo $response;
?>

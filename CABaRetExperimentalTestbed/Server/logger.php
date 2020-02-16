<?php
    $session = $_POST['videoSession'];
    $sessionID = $_POST['sessionID'];
    if($sessionID == ''){
        $a = array($session);
        $json = json_encode($a, JSON_PRETTY_PRINT );
        $filename = bin2hex(openssl_random_pseudo_bytes(4));
        file_put_contents( '../logfiles/'.$filename.'.json', $json);
        echo $filename;
    } else {
        $sessionID=preg_replace('/\s+/', '', $sessionID);
        $file_content = json_decode( file_get_contents('../logfiles/'.$sessionID.'.json') );
        array_push($file_content, $session);
        $file_content = json_encode ( $file_content, JSON_PRETTY_PRINT );
        file_put_contents( '../logfiles/'.$sessionID.'.json', $file_content );
        echo $sessionID;
    }
?>


package com.example.android.camera2basic;


import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.apache.http.HttpStatus;
import org.apache.http.entity.mime.HttpMultipartMode;
import org.apache.http.entity.mime.MultipartEntity;
import org.apache.http.entity.mime.content.ByteArrayBody;
import org.apache.http.entity.mime.content.ContentBody;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.CookieHandler;
import java.net.CookieManager;
import java.net.HttpURLConnection;
import java.net.URL;

import javax.net.ssl.SSLContext;

/**
 * Created by D038471 on 5/9/2016.
 */
public class Connection {

    private String BasicURL;
    public String sid;
    SSLContext sslContext = null;

    public void setBasicUrl(String url)
    {
        BasicURL = url;
    }

    Connection() {

        CookieManager cookieManager = new CookieManager();
        CookieHandler.setDefault(cookieManager);
    }

    public String doRequest(String ur)
    {
        return doRequest(ur, false, "");
    }

    public String doRequest(String ur, boolean bPost, String postCont)
    {
        HttpURLConnection urlConnection = null;
        String content = "";

        try {
            URL url = new URL(BasicURL + ur);

            urlConnection = (HttpURLConnection) url.openConnection();
            urlConnection.setRequestProperty ("User-Agent", "MyApp");

            if (bPost)
            {
                urlConnection.setRequestMethod("POST");
                if (!postCont.isEmpty())
                {
                    urlConnection.setDoOutput(true);
                    OutputStream out = new BufferedOutputStream(urlConnection.getOutputStream());
                    out.write(postCont.getBytes());
                    out.flush();
                    out.close();
                }
            }

            InputStream is = urlConnection.getInputStream();
            BufferedReader in = new BufferedReader(new InputStreamReader(is));
            String line = in.readLine();
            while (line != null)
            {
                content += line;
                line = in.readLine();
            }
        }
        catch(Exception e) {
            System.out.print(e.toString());
        }
        finally {
            if (urlConnection != null)
                urlConnection.disconnect();
        }

        return content;
    }

    private static String multipost(String urlString, MultipartEntity reqEntity) {
        try {
            URL url = new URL(urlString);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setReadTimeout(10000);
            conn.setConnectTimeout(15000);
            conn.setRequestMethod("POST");
            conn.setUseCaches(false);
            conn.setDoInput(true);
            conn.setDoOutput(true);

            conn.setRequestProperty("Connection", "Keep-Alive");
            conn.addRequestProperty("Content-length", reqEntity.getContentLength()+"");
            conn.addRequestProperty(reqEntity.getContentType().getName(), reqEntity.getContentType().getValue());

            OutputStream os = conn.getOutputStream();
            reqEntity.writeTo(conn.getOutputStream());
            os.close();
            conn.connect();

            int responseCode = conn.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) {
                return readStream(conn.getInputStream());
            }

        } catch (Exception e) {
            Log.e("DEBUG", "multipart post error " + e + "(" + urlString + ")");
        }
        return null;
    }

    private static String readStream(InputStream in) {
        BufferedReader reader = null;
        StringBuilder builder = new StringBuilder();
        try {
            reader = new BufferedReader(new InputStreamReader(in));
            String line = "";
            while ((line = reader.readLine()) != null) {
                builder.append(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return builder.toString();
    }


    public String test()
    {
        return doRequest("/");
    }

    public String uploadFile(Bitmap btmp, String file)
    {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        btmp.compress(Bitmap.CompressFormat.JPEG, 100, bos);
        ContentBody contentPart = new ByteArrayBody(bos.toByteArray(), file);
        MultipartEntity reqEntity = new MultipartEntity(HttpMultipartMode.BROWSER_COMPATIBLE);
        reqEntity.addPart("files", contentPart);
        return multipost(BasicURL + "/upload", reqEntity);
    }

    public Bitmap getBitmapFromURL(String src) {
        try {
            src = src.replace(" ", "%20");
            java.net.URL url = new java.net.URL(src);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestProperty ("User-Agent", "MyApp");
            connection.setRequestProperty ("content-type", "image/jpeg");
            connection.setRequestMethod("GET");
            connection.setDoInput(true);
            connection.setDoOutput(false);
            connection.connect();
            int status = connection.getResponseCode();
            InputStream input = null;

            if(status == HttpStatus.SC_OK)
            {
                input = connection.getInputStream();
                return BitmapFactory.decodeStream(input);
            }
            return null;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

}

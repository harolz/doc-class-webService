package com.example.demo.Controller;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.util.*;
import javax.annotation.PostConstruct;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.stereotype.Controller;

import javax.xml.bind.JAXBException;

import org.springframework.web.util.UriTemplate;
import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.FieldValue;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.xml.sax.SAXException;
import com.alibaba.fastjson.JSONObject;
import org.springframework.web.bind.annotation.*;
import org.jpmml.evaluator.*;
import com.example.demo.Util.LinkUtil;

@Controller
public class ServiceController {
    public static Evaluator evaluator;

    @PostConstruct
    public void init() {
        try {
            InputStream file = ServiceController.class.getClassLoader().getResourceAsStream("doc_class.pmml");
            evaluator = new LoadingModelEvaluatorBuilder()
                    .load(file)
                    .build();
            evaluator.verify();
        } catch (SAXException | JAXBException e) {
            e.printStackTrace();
        }
    }
    @RequestMapping(value = "/ping", method = RequestMethod.GET)
    public String ping() {
        return "";
    }

    @GetMapping("/")
    @ResponseStatus(value = HttpStatus.NO_CONTENT)
    public void discoverRoot(final HttpServletRequest request, final HttpServletResponse response) throws IOException {
        String rootUri = request.getRequestURL().toString();
        URI predictUri = new UriTemplate("{rootUri}{resource}").expand(rootUri, "predict");
        URI indexUri = new UriTemplate("{rootUri}{resource}").expand(rootUri, "index.html");
        String linkToIndex = LinkUtil.createLinkHeader(indexUri.toASCIIString(), "index");
        response.addHeader("Link", linkToIndex);
        String linkToPredict = LinkUtil.createLinkHeader(predictUri.toASCIIString(), "restconf");
        response.addHeader("Link", linkToPredict);
    }

    @RequestMapping(value= "/predict", method = RequestMethod.POST, produces = "application/json;charset=UTF-8")
    public  @ResponseBody JSONObject postPredict(@RequestBody String postData){
        // Parse String into JSON
        JSONObject json = JSONObject.parseObject(postData);
        // Predict using OCR words data
        String words = (String) json.get("words");
        JSONObject y = (JSONObject) predict(words);
        // return
        return y;
    }

    @GetMapping(value = "/predict/{words}", produces = "application/json;charset=UTF-8")
    public @ResponseBody JSONObject getPredict(@PathVariable String words) {
        System.out.println(words);
        JSONObject y = (JSONObject) predict(words);
        // return
        return y;
    }

    @RequestMapping("/")
    public String welcome(Map<String, Object> model) {
        return "index.html";
    }

    public static void print(Object... args){
        Arrays.stream(args).forEach(System.out::print);
        System.out.println("");
    }
    // define predict function
    // The param passed is a String of rawText input
    public static Object predict(String input){
        // Get InputField
        List<? extends InputField> inputFields = evaluator.getInputFields();
        print("InputField：", inputFields);
        // TargetField
        List<? extends TargetField> targetFields = evaluator.getTargetFields();
        print("TargetField：",targetFields);
        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();
        for(InputField inputField: inputFields){
            FieldName inputName = inputField.getName();
            FieldValue inputValue = inputField.prepare(input);
            arguments.put(inputName, inputValue);
        }

        // Predict using wrapped arguments
        Map<FieldName, ?> results = evaluator.evaluate(arguments);
        Map<String, ?> resultRecord = EvaluatorUtil.decode(results);
        String predictedType = (String) resultRecord.get("y");
        Classification classification;
        classification = (Classification) results.get(new FieldName("y"));
        Object value = classification.getValues().get(predictedType);
        JSONObject json = new JSONObject();
        json.put("result", predictedType);
        json.put("confidence",Double.parseDouble(value.toString()));
        return json;
    }


}





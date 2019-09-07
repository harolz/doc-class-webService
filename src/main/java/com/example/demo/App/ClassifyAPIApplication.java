package com.example.demo.App;

import com.example.demo.Filter.CORSFilter;
import com.example.demo.Filter.URLFilter;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;


@SpringBootConfiguration
@EnableAutoConfiguration
@ComponentScan(basePackages = "com.example.demo.Controller")
public class ClassifyAPIApplication {

    public static void main(String[] args) {
        serve();
    }

    public static void serve() {
        SpringApplication.run(ClassifyAPIApplication.class);
    }

    @Bean
    public FilterRegistrationBean commonsRequestLoggingFilter()
    {
        final FilterRegistrationBean registrationBean = new FilterRegistrationBean();
        registrationBean.setFilter(new CORSFilter());
        registrationBean.setFilter(new URLFilter());
        return registrationBean;
    }

}











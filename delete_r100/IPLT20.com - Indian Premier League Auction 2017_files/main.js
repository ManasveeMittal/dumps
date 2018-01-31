/*globals PULSE, PULSE.app */

(function( app ){
	"use strict";

	app.templates = {};
	app.common = {};
	app.defaultLanguage = "EN";
    app.account = 'ipl';

	app.paths = [
		{
			label: "local",
			domain: "localhost",
			// cdn: "../../",
			cdn: "dist/"
		},
        {
			label: "bsk",
			domain: "10\.0\.(4|5)\.([0-9])+",
			// cdn: "dist/"
			cdn: "../../"
		},
		{
			label: "development",
			canary: "https://api.canary.platform.pulselive.com/dev/",
			domain: "dev-ipl.pulselive.com",
			cmsAccount: 3,
            api: "//api.dev.platform.iplt20.com/",
            trackingAPI: "https://5f77vsjiq2.execute-api.us-east-1.amazonaws.com/dev/interactions",
			cdn: "/resources/ver/",
            playerImagePath: "//iplstatic.s3.amazonaws.com/players/2016/"
		},
		{
			label: "test",
			domain: "test-ipl.pulselive.com",
			canary: "https://api.canary.platform.pulselive.com/production/",
			cmsAccount: 11,
            api: "//api.test.platform.iplt20.com/",
            trackingAPI: "https://5f77vsjiq2.execute-api.us-east-1.amazonaws.com/test/interactions",
			cdn: "/resources/ver/",
            playerImagePath: "//iplstatic.s3.amazonaws.com/players/"
		},
		{
			label: "staging",
			domain: "stage-ipl.pulselive.com",
            api: "//api.platform.iplt20.com/",
            trackingAPI: "https://5f77vsjiq2.execute-api.us-east-1.amazonaws.com/prod/interactions",
			canary: "https://api.canary.platform.pulselive.com/production/",
			cmsAccount: 15,
			cdn: "/resources/ver/",
            playerImagePath: "//iplstatic.s3.amazonaws.com/players/"
		},
		{
			label: "production",
			canary: "https://api.canary.platform.pulselive.com/production/",
			domain: "iplt20.com",
            api: "//api.platform.iplt20.com/",
            trackingAPI: "https://5f77vsjiq2.execute-api.us-east-1.amazonaws.com/prod/interactions",
			cmsAccount: 15,
			cdn: "/resources/ver/",
            playerImagePath: "//iplstatic.s3.amazonaws.com/players/"
		}
	];

	app.checkEnvironment = function(){

		var location = window.location.hostname;
		var environment;

		app.paths.map( function( path ) {
			if( location === path.domain || location.match( new RegExp( path.domain ) ) ) {
				environment = path;
			}
		});

		return environment || "There are no app.paths defined for this domain";

	};

	app.environment = app.checkEnvironment();

}( PULSE.app ));

/*globals PULSE, PULSE.app, PULSE.core */


(function( app, core ){
	"use strict";

	app.I18N = {};

	app.I18N.setup = function(){
		var req_language = core.localStorage.getStorage( 'req_language', true );
	    app.language = req_language || app.defaultLanguage;
	    var Translator = new app.I18N.Translator( PULSE.I18N );

	    if (!PULSE.I18N) { PULSE.I18N = {}; }
	    PULSE.I18N.lookup = function()
	    {
	        return Translator.lookup.apply( Translator, arguments );
	    };
	    if( typeof window.moment !== 'undefined' )
	    {
	        app.I18N.enOverride();
	        moment.locale( app.language );
	    }
	};

	app.I18N.enOverride = function() {
	    moment.locale( 'en',
	    {
	        longDateFormat : {
	            LT: "HH:mm",
	            // LT: "h:mm A",
	            LTS: "h:mm:ss A",
	            l : 'DD/MM',
	            L: "DD/MM/YYYY",
	            ll: 'D MMMM',
	            LL: "D MMMM YYYY",
	            lll: "MMM D YYYY LT",
	            LLL: "MMMM Do YYYY LT",
	            llll: "ddd MMM D YYYY",
	            LLLL: "dddd, MMMM Do YYYY"
	        },
	        yearFirst: false
	    } );
	};


	app.I18N.Translator = function( translationsData )
	{
		var _self = this;

	    _self.hasTranslations = false;
	    _self.language = app.language;

	    if( translationsData )
	    {
	        _self.hasTranslations = true;
	        if( _self.language !== translationsData.language )
	        {
	            if( translationsData.language === undefined )
	            console.log( 'Language mismatch! Using ' + translationsData.language );
	            _self.language = translationsData.language;
	        }

	        _self.translations = translationsData.translations || {};
	        _self.hasTranslations = true;
	    }
	    else
	    {
	        _self.hasTranslations = false;
	        _self.translations = {};
	    }
	};

	app.I18N.Translator.prototype.lookup = function( key, replaceMap )
	{
		var _self = this;
	    if( key )
	    {
	        var mapping = _self.lookupKey( key );
	        if( replaceMap )
	        {
	            for( var replaceKey in replaceMap )
	            {
	                var regExp = new RegExp( "\\${" + replaceKey + "}", "gi" );
	                mapping = mapping.replace( regExp, replaceMap[ replaceKey ] );
	            }
	        }

	        return mapping;
	    }

	    return "";
	};

	app.I18N.Translator.prototype.lookupKey = function( key )
	{
		var _self = this;
	    if( _self.hasTranslations )
	    {
	        return _self.translations[ key ] || key;
	    }
	    else
	    {
	        return key;
	    }
	};


}( PULSE.app, PULSE.core ));

( function( app ){

    "use strict";

    /**
     * Map of media query size labels to screen widths
     * Mirrors what's in src/styles/layout/_mq.scss so the JS & CSS use the same values
     */
    app.measurements = {
        mobile: 400,
        phablet: 640,
        tablet: 840,
        desktop: 1025,
        wide: 1300
    };

}( PULSE.app ) );

( function( app, core, common, cricket ) {

    "use strict";

    app.templating = {};

    /**
     *  get generic date
     * @param {string} date
     * @return {string}
     */
    var getDateNice = function( date ) {
        var momentDate = moment( date ).utc();

        return momentDate.format("DD MMM YY");
    };

    app.templating.scoreToString = function( inningsSummaries ) {
        return inningsSummaries.map( function( summary ) {
            return cricket.utils.getInningsScore( summary.runs, summary.wkts, summary.allOut, summary.declared );
        } ).join( ' & ' );
    };

    /**
     * date age helper function
     * @param {string} date - expects date as string
     * @return {string}
     */
    var dateAge = function( date ) {

        var ONE_MINUTE = 1000 * 60;
        var ONE_HOUR = 60 * ONE_MINUTE;
        var ONE_DAY = 24 * ONE_HOUR;

        var output = '';

        if ( date ) {
            date = typeof date === 'string' ? core.date.parseString( date ) : new Date( date );
            date = moment( date ).utc();

            var diff = moment().format('x') - date;

            var dateAge = 'old';
            if ( diff < ONE_DAY ) {
                dateAge = 'new';
            }

            var current, hours, mins, hoursLabel, minsLabel;
            current = diff;
            hours = Math.floor( current / ONE_HOUR );
            mins = Math.round( (current / ONE_MINUTE ) % 60 );

            // output difference of current date and published date
            if ( hours > 0 ) {
                hoursLabel = hours === 1 ? 'hr' : 'hrs';
                output = hours + ' ' + hoursLabel + ' ';
            }
            minsLabel = mins === 1 ? 'min' : 'mins';
            output += mins + ' ' + minsLabel + ' ago';

            // if date is old override output
            if ( dateAge === 'old' ) {
                output = getDateNice( new Date( date ) );
            }

            return output;
        }
        return ''; // return nothing in false case
    };

    /**
     * Parses an HTML string to an actual element
     *
     * @param {String} htmlString - HTML string you want to parse to an HTML element
     */
    var stringToElement = function( htmlString ) {

        var d = document.createElement( 'div' );
        d.innerHTML = htmlString.trim();
        return d.firstChild;
    };

    var scoreToString = function( inningsSummaries ) {
        return inningsSummaries.map( function( summary ) {
            return cricket.utils.getInningsScore( summary.runs, summary.wkts, summary.allOut, summary.declared );
        } ).join( ' & ' );
    };

    var getFormattedDate = function( match, timezone, dateFormat ) {
        var date = match.getDate();
        if( date ) {
            var momentDate = moment( date );
            switch( timezone ) {
                case 'BST':
                    return momentDate.utc().add( 1, 'hours' ).format( dateFormat );
                case 'local':
                    return moment( date ).utc().add( match.timezoneOffset, 'hours' ).format( dateFormat );
                // case 'GMT':
                default:
                    return momentDate.utc().format( dateFormat );

            }
        }
        return '';
    };

    var buildDescription = function() {
        var description = '';
        if( arguments && arguments.length ) {
            description = Array.prototype.filter.call( arguments, function( arg ) {
                return typeof arg !== 'undefined' && arg;
            } ).join( ', ' );
        }
        return description;
    };

    var getVenueString = function( venue ) {
        var venueString = 'TBC';
        if( venue && venue.fullName !== 'TBC' ) {
            venueString = venue.shortName || venue.fullName;
            if( venue.city ) {
                venueString += ', ' + venue.city;
            }
        }
        return venueString;
    };

    var getMatchTypeLabel = function( matchType ) {
        switch( matchType ) {
            case 'TEST':
                return 'Test';
            case 'FIRST_CLASS':
                return 'First Class';
            case 'LIST_A':
                return 'List A';
            case 'ODI':
                return 'ODI';
            case 'T20':
                return 'T20';
            case 'T20I':
                return 'T20I';
            case 'WODI':
                return 'ODI';
            case 'WT20':
                return 'T20';
            case 'WT20I':
                return 'T20I';
            default:
                return matchType;
        }
    };

    /**
     * Can only be used with Scoring Publication data, returns a user-friendly string for the
     * description of the innings (e.g., "England 1st Innings")
     * @param  {cricket.Match} match - the match
     * @param  {Number} inningsIndex - the index of the innings
     * @return {String}              - the user-facing string
     */
    var getInningsLabel = function( match, inningsIndex ) {
        if( match && match.getBattingOrder().length ) {
            inningsIndex = inningsIndex > -1 ? inningsIndex : ( match.currentState.currentInningsIndex || 0 );
            var team = match.getTeam( match.getBattingOrder()[ inningsIndex ] );
            if( match.isLimitedOvers() ) {
                return ( team.shortName || team.fullName ) + ' Innings';
            }
            else {
                var inningsOrdinal = inningsIndex > 1 ? ' 2nd ' : ' 1st ';
                return ( team.shortName || team.fullName ) + inningsOrdinal + 'Innings';
            }
        }
        return '';
    };

    var getPlayerHeadshotUrl = function( playerId, matchType, extension, size ) {

        extension = extension || 'png';
        size = size || '480x480';

        switch( matchType ) {
            case 'TEST':
                return "https://icc-resources.s3.amazonaws.com/player-photos/test/" + size + "/" + playerId + "." + extension;
            case 'ODI':
                return "https://icc-resources.s3.amazonaws.com/player-photos/odi/" + size + "/" + playerId + "." + extension;
            case 'T20I':
                return "https://icc-resources.s3.amazonaws.com/player-photos/t20i/" + size + "/" + playerId + "." + extension;
            default:
                return "https://icc-resources.s3.amazonaws.com/player-photos/test/" + size + "/photo-missing.png";
        }
    };

    var getMatchDateRange = function( match, format ) {

        var dateFormatShort = format || "ddd D";
        var dateFormat = format || "ddd D MMMM";


        if( match.getEndDate && match.getDate().getDate() != match.getEndDate().getDate() ) {
            if( match.getDate().getMonth() === match.getEndDate().getMonth() ){
                return moment( match.getDate() ).format( dateFormatShort ) + " - " +  moment( match.getEndDate() ).format( dateFormat );
            }
            else{
                return moment( match.getDate() ).format( dateFormat ) + " - " +  moment( match.getEndDate() ).format( dateFormat );
            }
        }
        else{
            return moment( match.getDate() ).format( dateFormat );
        }

    };

    var getDateDiff = function (start, end, measurement ) {
        measurement = measurement || 'days';
        var startDate = moment(start);
        var endDate = moment(end);
        return endDate.diff(startDate, measurement);
    };

    /**
	 * Get duration in time format mm:ss
	 *
	 * @param {Int} number of seconds
	 * @returns {String} output duration in format mm:ss
	 */
	var durationToTime = function( duration ) {
        var sec_num = parseInt(duration, 10);

        if (sec_num) {
            var hours   = Math.floor(sec_num / 3600);
            var minutes = Math.floor((sec_num - (hours * 3600)) / 60);
            var seconds = sec_num - (hours * 3600) - (minutes * 60);

            if (hours   < 10) { hours   = "0" + hours; }
            if (minutes < 10) { minutes = "0" + minutes; }
            if (seconds < 10) { seconds = "0" + seconds; }

            var minSec = minutes + ':' + seconds;

            return hours > 0 ? hours + ':' + minSec : minSec;
        }

        return '00:00';
    };

    /**
     * Helper for pluralisation of nouns
     * @param {Number} number - the number to base the logic off of
     * @param {String} singular - the singular version of the noun
     * @param {String} plural - the plural version of the noun
     * @param {Boolean} includeNumber - whether to prepend the number to the output string or not
     */
    var pluralise = function( number, singular, plural, includeNumber ) {
        var string = includeNumber ? number + ' ' : '';
        return string + ( number == 1 ? singular : plural );
    };

    /**
     * object with all helper functions for underscore templates
     */
    app.templating.helper = {
        dateAge: dateAge,
        getPlayerHeadshotUrl: getPlayerHeadshotUrl,
        getDateDiff: getDateDiff,
        durationToTime: durationToTime,
        buildDescription: buildDescription,
        pluralise: pluralise,
        cricket: {
            getInningsLabel: getInningsLabel,
            getMatchTypeLabel: getMatchTypeLabel,
            getVenueString: getVenueString,
            scoreToString: scoreToString,
            getFormattedDate: getFormattedDate,
            getMatchDateRange: getMatchDateRange
        }
    };

    /**
     * Renders a template with the given data and returns the compiled template
     *
     * @param {Object}  data              - data to render in JSON format
     * @param {String}  templateName      - the name of the template (must match file name)
     * @param {Boolean} parseToHtmlObject - parse the rendered template string to an HTML object - default false
     * @return {(String|DOMElement)}      - Rendered template with model
     */
    app.templating.render = function( model, templateName, parseToHtmlObject ) {
        var renderedTemplate = '';

        model = model || {};
        for( var func in app.templating ) {
            if( app.templating.hasOwnProperty( func ) ) {
                model[ func ] = app.templating[ func ];
                model.urlUtil = {
                    generateUrl: app.common.generateUrl
                };
                model.contentUtil = app.common.content;
                model.imageUtil = app.common.image;
            }
        }

        if( templateName ) {
            var templateEngine = app.templates[ templateName ];
            if( templateEngine ) {
                try {
                    renderedTemplate = templateEngine( model );
                }
                catch( e ) {
                    if( e.message ) {
                        e.message += ' in template ' + templateName;
                    }
                    console.warn( e );
                }
                if( parseToHtmlObject ) {
                    return stringToElement( renderedTemplate );
                }
            }
            else {
                console.warn( 'No template was rendered. Template not found: ' + templateName );
            }
        }
        return renderedTemplate;
    };

}( PULSE.app, PULSE.core, PULSE.app.common, PULSE.cricket ) );

/*globals PULSE, PULSE.app */

(function( app ){
	"use strict";

	app.widgetDeps = function(){

		var environment = app.checkEnvironment();
		var els = document.querySelectorAll( '[data-script]' );
		var deps = [];
		var scriptPrefix = app.environment.cdn + "widgets/";
		var scriptSuffix = app.environment.label === "production" ? ".min.js" : ".js";

		Array.prototype.map.call( els, function( el ){
            el.getAttribute( 'data-script' ).split(',').forEach( function( s ) {
                s = s ? s.trim() : undefined;
                if( s ) {
                    addDependency( s );
                }
            } );
		} );

		function addDependency( dep ){
			if( deps.indexOf( dep ) < 0 ){
				deps.push( dep );
				var script = document.createElement( 'script' );
				script.type = 'text/javascript';
				script.src = scriptPrefix + dep + scriptSuffix;
				document.body.appendChild( script );
			}
		}
	};

}( PULSE.app ));

/*globals PULSE, PULSE.app */

(function( app ){
	"use strict";

	window.onload = function(){
		app.widgetDeps();
		app.I18N.setup();

		/** If FastClick.js is loaded it rebinds all click events with touch events where necessary */
		if( FastClick !== null ){
			FastClick.attach(document.body);
		}
	};

}( PULSE.app ));